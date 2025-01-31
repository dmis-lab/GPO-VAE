from typing import Dict, Iterable, Literal, Optional, Sequence, Union, List, Tuple
from datetime import datetime
from pytz import timezone
from tqdm.auto import tqdm
import gc, random, pickle

import scipy
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader

import anndata
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp

from gpo_vae.analysis.average_treatment_effects import (
    estimate_model_average_treatment_effect,
)
from gpo_vae.data.utils.perturbation_datamodule import PerturbationDataModule
from gpo_vae.data.utils.perturbation_dataset import PerturbationDataset

class PerturbationPlatedPredictor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        guide: nn.Module,
        local_variables: Optional[Iterable[str]] = None,
        perturbation_plated_variables: Optional[Iterable[str]] = None,
        dosage_independent_variables: Optional[Iterable[str]] = None,
    ):
        super().__init__()

        # convert variables to lists
        local_variables = list(local_variables) if local_variables is not None else []
        perturbation_plated_variables = (
            list(perturbation_plated_variables)
            if perturbation_plated_variables is not None
            else []
        )

        # check valid variable lists
        assert sorted(model.get_var_keys()) == sorted(
            guide.get_var_keys()
        ), "Mismatch in model and guide variables"

        # make sure that all variables are specified as a local variable or a
        # perturbation plated variable
        variables = local_variables + perturbation_plated_variables
        assert sorted(list(model.get_var_keys())) == sorted(
            variables
        ), "Mismatch between model variables and variables specified to loss module"

        # make sure that dosage_independent_variables are valid
        if dosage_independent_variables is not None:
            assert set(dosage_independent_variables).issubset(set(variables))

        # store passed in values
        self.model = model.eval()
        self.guide = guide.eval()
        self.local_variables = local_variables
        self.perturbation_plated_variables = perturbation_plated_variables
        self.dosage_independent_variables = dosage_independent_variables
        
        '''
        '''
        n_phenos = guide.n_phenos
        

    def _get_device(self):
        # TODO: clean up device management approach
        # assumes all parameters/buffers for model and guide are on same device
        device = next(self.model.parameters()).device
        return device

    @torch.no_grad()
    def compute_predictive_iwelbo(
        self,
        loaders: Union[DataLoader, Sequence[DataLoader]],
        n_particles: int,
    ) -> pd.DataFrame:
        """
        Compute IWELBO(X|variables, theta, phi) for trained model
        Importantly, does not score plated variables against priors

        Parameters
        ----------
        loaders: dataloaders with perturbation datasets
        n_particles: number of particles to compute predictive IWELBO

        Returns
        -------
        Dataframe with estimated predictive IWELBO for each datapoint
        in column "IWELBO", sample IDs in index

        """
        if isinstance(loaders, DataLoader):
            loaders = [loaders]

        device = self._get_device()

        # sample perturbation plated variables to share across batches
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        condition_values = {}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]

        # compute importance weighted ELBO
        id_list = []
        iwelbo_list = []
        for loader in loaders:
            idx_list_curr = []
            for batch in tqdm(loader):
                for k in batch:
                    batch[k] = batch[k].to(device)
                idx_list_curr.append(batch["idx"].detach().cpu().numpy())

                # catch adding library size if it becomes relevant
                # note: this part is not necessary for the guide
                # typically the llk is not evaluated in the guide, so we can skip this
                if self.model.likelihood_key == "library_nb":
                    condition_values["library_size"] = batch["library_size"]

                guide_dists, guide_samples = self.guide(
                    X=batch["X"],
                    D=batch["D"],
                    qc=batch["qc"],
                    condition_values=condition_values,
                    n_particles=n_particles,
                )

                # catch adding library size if it becomes relevant to the likelihood
                # necessary to evaluate predictive
                # this is strictly not the elegant way to do this, that would
                # be via args/kwargs, but a quick fix
                if self.model.likelihood_key == "library_nb":
                    guide_samples["library_size"] = batch["library_size"]

                if "qm" in self.model._get_name():
                    model_dists, model_samples = self.model(
                        D=batch["D"],
                        qc=batch["qc"],
                        condition_values=guide_samples,
                        n_particles=n_particles,
                    )
                else:
                    model_dists, model_samples = self.model(
                        D=batch["D"],
                        qc=batch["qc"],
                        condition_values=guide_samples,
                        n_particles=n_particles,
                    )

                iwelbo_terms_dict = {}
                # shape: (n_particles, n_samples)
                # iwelbo_terms_dict["x"] = model_dists["p_x"].log_prob(batch["X"]).sum(-1)
                p_x_good_log_p = model_dists['p_x_good'].log_prob(batch["X"])
                p_x_bad_log_p = model_dists['p_x_bad'].log_prob(batch["X"])
                if batch["qc"].shape[1] > 1:                    
                    b_qc = batch["qc"].max(axis=1)[0].unsqueeze(-1)
                    p_x_real_log_p = ((1-b_qc)*p_x_good_log_p) + (b_qc*p_x_bad_log_p)
                else:
                    p_x_real_log_p = ((1-batch["qc"])*p_x_good_log_p) + (batch["qc"]*p_x_bad_log_p)
                iwelbo_terms_dict["x"] = p_x_real_log_p.sum(-1)
                for var_name in self.local_variables:
                    p = (
                        model_dists[f"p_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    q = (
                        guide_dists[f"q_{var_name}"]
                        .log_prob(guide_samples[var_name])
                        .sum(-1)
                    )
                    iwelbo_terms_dict[var_name] = p - q

                # shape: (n_particles, n_samples)
                iwelbo_terms = sum([v for k, v in iwelbo_terms_dict.items()])
                # compute batch IWELBO
                # shape: (n_samples,)
                batch_iwelbo = torch.logsumexp(iwelbo_terms, dim=0) - np.log(
                    n_particles
                )

                iwelbo_list.append(batch_iwelbo.detach().cpu().numpy())

            idx_curr = np.concatenate(idx_list_curr)
            dataset: PerturbationDataset = loader.dataset
            ids_curr = dataset.convert_idx_to_ids(idx_curr)
            id_list.append(ids_curr)

        iwelbo = np.concatenate(iwelbo_list)
        ids = np.concatenate(id_list)

        iwelbo_df = pd.DataFrame(
            index=ids, columns=["IWELBO"], data=iwelbo.reshape(-1, 1)
        )
        return iwelbo_df

    @torch.no_grad()
    def sample_observations(
        self,
        dosages: torch.Tensor,
        perturbation_names: Optional[Sequence[str]],
        n_particles: int = 1,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        x_var_info: Optional[pd.DataFrame] = None,
        qc=None
    ) -> anndata.AnnData:
        """
        Sample observations conditioned on perturbations

        Parameters
        ----------
        dosages: encoded dosages for perturbations of interest
        perturbation_names: optional string names for each row in dosages
        n_particles: number of samples to take for each dosage

        Returns
        -------
        anndata of samples dosage index, perturbation name, and particle_idx in obs,
        sampled observations in X, and x_var_info in var
        """
        device = self._get_device()
        dosages = dosages.to(device)
        qc = qc.to(device)
        # sample perturbation plated variables to share across batches
        guide_dists, guide_samples = self.guide(n_particles=n_particles)
        if condition_values is None:
            condition_values = dict()
        else:
            condition_values = {k: v.to(device) for k, v in condition_values.items()}
        for var_name in self.perturbation_plated_variables:
            condition_values[var_name] = guide_samples[var_name]

        x_good_samples_list, x_bad_samples_list = [], []
        for i in tqdm(range(dosages.shape[0])):
            D = dosages[i : i + 1]
            _, model_samples = self.model(
                D=D, condition_values=condition_values, n_particles=n_particles, qc=qc,
            )
            x_good_samples_list.append(model_samples["x_good"].detach().cpu().numpy().squeeze())
            x_bad_samples_list.append(model_samples["x_bad"].detach().cpu().numpy().squeeze())
            

        x_good_samples = np.concatenate(x_good_samples_list)
        obs = pd.DataFrame(index=np.arange(x_good_samples.shape[0]))
        obs["perturbation_idx"] = np.repeat(np.arange(dosages.shape[0]), n_particles)
        obs["particle_idx"] = np.tile(np.arange(dosages.shape[0]), n_particles)
        if perturbation_names is not None:
            obs["perturbation_name"] = np.array(perturbation_names)[
                obs["perturbation_idx"].to_numpy()
            ]

        adata_good = anndata.AnnData(obs=obs, X=x_good_samples)
        if x_var_info is not None:
            adata_good.var = x_var_info.copy()

        x_bad_samples = np.concatenate(x_bad_samples_list)
        obs = pd.DataFrame(index=np.arange(x_bad_samples.shape[0]))
        obs["perturbation_idx"] = np.repeat(np.arange(dosages.shape[0]), n_particles)
        obs["particle_idx"] = np.tile(np.arange(dosages.shape[0]), n_particles)
        if perturbation_names is not None:
            obs["perturbation_name"] = np.array(perturbation_names)[
                obs["perturbation_idx"].to_numpy()
            ]

        adata_bad = anndata.AnnData(obs=obs, X=x_bad_samples)
        if x_var_info is not None:
            adata_bad.var = x_var_info.copy()
        return adata_good, adata_bad

    def sample_observations_data_module(
        self,
        data_module: PerturbationDataModule,
        n_particles: int,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        qc = None,
        additional_D = False,
    ):
        """
        Sample observations from each unique intervention observed in a PerturbationDataModule
        TODO: come up with better naming for this method

        Parameters
        ----------
        data_module
        n_particles

        Returns
        -------
        anndata with samples from unique interventions in data module
        obs will have perturabtion name and particle idx, X will have sampled observations,
        and var dataframe will have
        """
        if additional_D:
            perturbation_names = data_module.get_d_var_info()
        else:
            perturbation_names = data_module.get_unique_observed_intervention_info().index
        D = data_module.get_unique_observed_intervention_dosages(perturbation_names)
        x_var_info = data_module.get_x_var_info()

        adata_good, adata_bad = self.sample_observations(
            dosages=D,
            perturbation_names=perturbation_names,
            x_var_info=x_var_info,
            n_particles=n_particles,
            condition_values=condition_values,
            qc=qc
        )

        return adata_good, adata_bad

    @torch.no_grad()
    def estimate_average_treatment_effects(
        self,
        dosages_alt: torch.Tensor,
        dosages_control: torch.Tensor,
        quality: torch.Tensor,
        method: Literal["mean", "perturbseq"],
        n_particles: int = 1000,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        perturbation_names_alt: Optional[Sequence[str]] = None,
        perturbation_name_control: Optional[str] = None,
        x_var_info: Optional[pd.DataFrame] = None,
        batch_size: int = 500,
    ) -> anndata.AnnData:
        """
        Estimate average treatment effects of alternate dosages relative control dosage using model

        Parameters
        ----------
        dosages_alt: alternate dosages
        dosages_control: control dosage
        method: mean or perturbseq (log fold change after normalization for library size)
        n_particles: number of samples per treatment for estimate
        condition_values: any additional conditioning variables for model / guide
        perturbation_names: names for dosages, will be used as obs index if provided
        x_var_info: names of observed variables, will be included as var if provided

        Returns
        -------
        anndata with average treatment effects in X, perturbation names as obs index if provided
        (aligned to dosages_alt otherwise), and x_var_info as var if provided
        """
        device = self._get_device()
        dosages_alt = dosages_alt.to(device)
        dosages_control = dosages_control.to(device)
        quality = quality.to(device)
        if condition_values is not None:
            for k in condition_values:
                condition_values[k] = condition_values[k].to(device)

        average_treatment_effects = estimate_model_average_treatment_effect(
            model=self.model,
            guide=self.guide,
            dosages_alt=dosages_alt,
            dosages_control=dosages_control,
            quality=quality,
            n_particles=n_particles,
            method=method,
            condition_values=condition_values,
            batch_size=batch_size,
            dosage_independent_variables=self.dosage_independent_variables,
        )
        adata = anndata.AnnData(average_treatment_effects)
        if perturbation_names_alt is not None:
            adata.obs = pd.DataFrame(index=np.array(perturbation_names_alt))
        if perturbation_name_control is not None:
            adata.uns["control"] = perturbation_name_control
        if x_var_info is not None:
            adata.var = x_var_info.copy()
        return adata

    def estimate_average_effects_data_module(
        self,
        data_module: PerturbationDataModule,
        control_label: str,
        method: Literal["mean", "perturbseq"],
        n_particles: int = 100,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 500,
    ):
        perturbation_names = data_module.get_unique_observed_intervention_info().index
        perturbation_names_alt = [
            name for name in perturbation_names if name != control_label
        ]

        dosages_alt = data_module.get_unique_observed_intervention_dosages(
            perturbation_names_alt
        )
        dosages_ref = data_module.get_unique_observed_intervention_dosages(
            [control_label]
        )
        # good_quality = torch.zeros((1, data_module.get_qc_var_info().shape[0]))
        good_quality = torch.ones((1, data_module.get_qc_var_info().shape[0]))

        x_var_info = data_module.get_x_var_info()

        adata = self.estimate_average_treatment_effects(
            dosages_alt=dosages_alt,
            dosages_control=dosages_ref,
            quality=good_quality,
            method=method,
            n_particles=n_particles,
            condition_values=condition_values,
            perturbation_names_alt=perturbation_names_alt,
            perturbation_name_control=control_label,
            x_var_info=x_var_info,
            batch_size=batch_size,
        )
        return adata

    def estimate_qc_ratio(
        self,
        data_module: PerturbationDataModule,
        n_particles: int,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        thr: int = 3,
    ):
        good_quality = torch.zeros((1, data_module.get_qc_var_info().shape[0]))
        # good_quality = torch.ones((1, data_module.get_qc_var_info().shape[0]))
        adata, adata_bad = self.sample_observations_data_module(data_module,100,condition_values, good_quality)
        sub_adata_list = []
        
        n_each_pert = adata.shape[0]/len(adata.obs.perturbation_idx.unique())
        step = int(n_each_pert/10)
        for i in tqdm(range(10)):
            idx = adata.obs.groupby('perturbation_idx').apply(lambda x: x[i*step:(i+1)*step]).index.get_level_values(1)
            sub_adata = adata[idx]
            sub_adata = get_qc_annotation(sub_adata)
            sub_adata_list.append(sub_adata)
        adata = anndata.concat(sub_adata_list, axis=0)

        result = {}
        for thr in [3,4,5]:
            adata, qc_col = get_qc_one_hot_cols(adata, thr, data_module.thr_values[thr])
            qc_pass,qc_fail = adata.obs['total_qc'].value_counts()
            result[thr] = round(qc_pass / (qc_pass+qc_fail),4)
        return result





    def statistical_evaluation(
            self,
            data_module: PerturbationDataModule,
            network: List[Tuple],
            n_particles: int = 100,
            condition_values: Optional[Dict[str, torch.Tensor]] = None,
            max_path_length = 3,
            check_false_omission_rate=False,
            omission_estimation_size=0, 
            p_value_threshold=0.05,      
    ):
    
        good_quality = torch.zeros((1, data_module.get_qc_var_info().shape[0]))
        # adata, adata_bad = self.sample_observations_data_module(data_module, n_particles, condition_values, good_quality, additional_D=False)
        # adata = data_module.adata
        adata = data_module.adata[data_module.adata.obs['split'] == 'test']
        # self.expression_matrix = adata.X
        # self.gene_names = adata.var.index.to_list()
        # self.interventions = adata.obs['gene']
        self.expression_matrix, self.gene_names, self.interventions = preprocess_dataset(adata, criteria='FC3')
        self.gene_to_index = dict(zip(self.gene_names, range(len(self.gene_names))))
        self.index_to_gene = dict(zip(range(len(self.gene_names)), self.gene_names))
        
        self.gene_to_interventions = dict()
        for i, intervention in enumerate(self.interventions):
            self.gene_to_interventions.setdefault(intervention, []).append(i)
        statistical_evaluation = self.evaluate_network(network, 
                                                       max_path_length, 
                                                       check_false_omission_rate, 
                                                       omission_estimation_size, 
                                                       p_value_threshold)

        return statistical_evaluation


    def evaluate_network(
        self, 
        network: List[Tuple],
        max_path_length = 3, 
        check_false_omission_rate=False, 
        omission_estimation_size=0, 
        p_value_threshold=0.05,
    ) -> Dict:
        """
        Use a non-parametric Mannwhitney rank-sum test to test wether perturbing an upstream does have
        an effect on the downstream children genes. The assumptions is that intervening on a parent gene
        should have an effect on the distribution of the expression of a child gene in the network. Also consider
        the all connected graph with the same evaluation (all pairs such that there is a directed path between them)

        Args:
            network: output network as a list of tuples, where (A, B) indicates that gene A acts on gene B
            max_path_length: maximum length of paths to consider for evaluation. If -1, check paths of any length.
            check_false_omission_rate: whether to check the false omission rate of the predicted graph. 
                                        The false omission rate is defined as (FN / (FN + TN)), FN = false negative and TN = true negative
            omission_estimation_size: how many negative pairs (a pair predicted to have no interaction, i.e, there is no path in the output graph) to draw to estimate the false omission rate

        Returns:
            Number of true positive and false positive edges for both original graph and all connected graph
        """
        network_as_dict = {}
        pert_gene_num = len(self.gene_to_interventions)-1
        for a, b in network:
            if a >= pert_gene_num or b >= pert_gene_num:
                continue
            network_as_dict.setdefault(a, set()).add(b)
        true_positive, false_positive, wasserstein_distances = self._evaluate_network(
            network_as_dict, p_value_threshold
        )

        # all_connected_network = {**network_as_dict}
        all_connected_network = {}
        for a, b in network:
            all_connected_network.setdefault(a, set()).add(b)
        # Test graph with paths of length smaller or equal to max_path_length
        all_path_results = []
        if max_path_length == -1:
            max_path_length = len(self.gene_names)
        for _ in range(max_path_length - 1):
            single_step_deeper_all_connected_network = {
                v: n.union(
                    *[
                        all_connected_network[nn]
                        for nn in n
                        if nn in all_connected_network
                    ]
                )
                for v, n in all_connected_network.items()
            }
            single_step_deeper_all_connected_network = {v: c - {v} for v, c in single_step_deeper_all_connected_network.items()}
            if single_step_deeper_all_connected_network == all_connected_network:
                break
            new_edges = {
                key: single_step_deeper_all_connected_network[key] - all_connected_network[key]
                for key in single_step_deeper_all_connected_network
            }
            for k in list(new_edges.keys()):
                if k >= pert_gene_num:
                    del new_edges[k]
                    continue
                new_edges[k] = {v for v in new_edges[k] if v <= pert_gene_num}
            true_positive_connected, false_positive_connected, wasserstein_distances_connected = \
                self._evaluate_network(new_edges, p_value_threshold)
            all_path_results.append({
                "true_positives": true_positive_connected,
                "false_positives": false_positive_connected,
                "wasserstein_distance": {
                    "mean": np.mean(wasserstein_distances_connected)
                },
            })
            all_connected_network = single_step_deeper_all_connected_network
            
        if check_false_omission_rate:
            print("Check False Omission Rate")
            edges = set()
            # Draw omission_estimation_size edges from the negative set (edges predicted to have no interaction)
            # to estimate the false omission rate and the associated mean wasserstein distance
            while len(edges) < omission_estimation_size:
                pair = random.sample(range(len(self.gene_names)), 2)
                edge = self.gene_names[pair[0]], self.gene_names[pair[1]]
                if edge[0] in all_connected_network and edge[1] in all_connected_network[edge[0]]:
                    continue
                edges.add(edge)
            network_as_dict = {}
            for a, b in edges:
                if a not in self.gene_to_interventions.keys() or b not in self.gene_to_interventions.keys():
                    continue
                network_as_dict.setdefault(a, set()).add(b)
            res_random = self._evaluate_network(network_as_dict, p_value_threshold)
            false_omission_rate = res_random[0] / omission_estimation_size
            negative_mean_wasserstein = np.mean(res_random[2])
        else:
            false_omission_rate = -1
            negative_mean_wasserstein = -1

        return {
            "output_graph": {
                "true_positives": true_positive,
                "false_positives": false_positive,
                "wasserstein_distance": {"mean": np.mean(wasserstein_distances)},
            },
            "all_path_results": all_path_results,
            "false_omission_rate": false_omission_rate,
            "negative_mean_wasserstein": negative_mean_wasserstein
        }


    def _evaluate_network(self, network_as_dict, p_value_threshold):
        true_positive = 0
        false_positive = 0
        wasserstein_distances = []
        for parent in tqdm(network_as_dict.keys()):
            children = network_as_dict[parent]
            for child in children:
                observational_samples = self.get_observational(child)
                interventional_samples = self.get_interventional(child, parent)
                ranksum_result = scipy.stats.mannwhitneyu(
                    observational_samples, interventional_samples
                )
                wasserstein_distance = scipy.stats.wasserstein_distance(
                    observational_samples, interventional_samples, 
                )
                wasserstein_distances.append(wasserstein_distance)
                p_value = ranksum_result[1]
                if p_value < p_value_threshold:
                    # Mannwhitney test rejects the hypothesis that the two distributions are similar
                    # -> parent has an effect on the child
                    true_positive += 1
                else:
                    false_positive += 1
        return true_positive, false_positive, wasserstein_distances


    def get_observational(
        self, 
        child: str,
    ) -> np.array:
            """
            Return all the samples for gene "child" in cells where there was no perturbations
            Args:
                child: Gene name of child to get samples for
            Returns:
                np.array matrix of corresponding samples
            """
            return self.get_interventional(child, "non-targeting")


    def get_interventional(
        self, 
        child: str, 
        parent: str,
    ) -> np.array:
        """
        Return all the samples for gene "child" in cells where "parent" was perturbed
        Args:
            child: Gene name of child to get samples for
            parent: Gene name of gene that must have been perturbed
        Returns:
            np.array matrix of corresponding samples
        """
        if type(child) == str:
            child = self.gene_to_index[child]

        if type(parent) == str:
            parent = self.gene_to_interventions[parent]
        else:
            parent = self.gene_to_interventions[self.index_to_gene[parent]]

        return self.expression_matrix[parent, child]

    def load_gt(self, bioeval_dir_path, name="all"):
        if name == "all":
            gt_dict = {}
            for network_name in ["biogrid", "corum", "regnetwork", "string"]:
                gt_dict[network_name] = list(pickle.load(open(f"{bioeval_dir_path}/{network_name}.pkl", "rb")))
        else:
            raise "Input eval dataset name!"
        
        gt_dict["pool"] = list(set().union(*gt_dict.values()))
        return gt_dict


    def bio_network_preprop(self, gt_dict, map_dict):
        gt_networks_dict = {}
        for name, edge_list in gt_dict.items():
            if edge_list[0][0][0:4] != 'ENSG':
                true_network = [(map_dict[stt], map_dict[end]) for stt, end in edge_list if (stt in map_dict) and (end in map_dict)]
            else:
                true_network = [(stt, end) for stt, end in edge_list if (stt in map_dict.values()) and (end in map_dict.values())]
            gt_networks_dict[name] = true_network
        return gt_networks_dict


    def biological_evaluation(self, true_network_dict, predicted_network):
        result = dict(dict())
        for name, true_network in true_network_dict.items():
            predicted_network = set(predicted_network)
            true_network = set(true_network)
            total = predicted_network.union(true_network)
            pred_graph_total = [1 if x in predicted_network else 0 for x in total]
            true_graph_total = [1 if x in true_network else 0 for x in total]   
            precision = precision_score(true_graph_total, pred_graph_total, pos_label=1)
            recall = recall_score(true_graph_total, pred_graph_total, pos_label=1)
            result[f"{name}_precision"] = precision
            result[f"{name}_recall"] = recall
        return result



def preprocess_dataset(adata, perturbed_only = False, criteria = 'highly_variable'):
    """Preprocess the Anndata dataset and extract the necessary information
    Args:
        dataset_path (string): path to Anndata dataset
        summary_stats (pandas.DataFrame): dataframe containing summary stats to filter for strong perturbations. If None, do not filter.
    Returns:
        (numpy, list, list): expression matrix, list of gene ids (columns), list of perturbed genes (rows)
    """
    # Normalize data
    sc.pp.normalize_per_cell(adata, key_n_counts='UMI_count')
    sc.pp.log1p(adata)
    adata.obs["gene_id"] = adata.obs["gene"]
    # data_expr_raw.obs["gene_id"] = data_expr_raw.obs["perturbation_name"].apply(lambda x: x.split('_')[0])

    intervened_genes = list(adata.obs["gene_id"])
    expression_gene = list(adata.var.index) + ['non-targeting']
    gene_to_interventions_ser = adata[adata.obs['gene_id'].isin(expression_gene)].obs.reset_index().reset_index().groupby('gene_id')['index'].apply(list)
    intervened_genes_set = set(gene_to_interventions_ser.index)
    intervened_genes = ["excluded" if gene not in intervened_genes_set else gene for gene in intervened_genes]
    adata.obs['gene'] = intervened_genes
    adata = adata[adata.obs['gene'] != "excluded"]

    if perturbed_only:
        adata = adata[:,adata.var.index.isin(intervened_genes_set)]
    else:
        hvg = adata.var[adata.var[criteria]].index
        additional_dosage_ls = sorted(set(hvg) - set(adata.obs['gene']))
        isin_dosage_ls = sorted(set(adata.obs['gene']).intersection(set(adata.var.index)))
        adata = adata[:,isin_dosage_ls + additional_dosage_ls]

    return adata.X, adata.var.index.to_list(), adata.obs['gene']


    intervened_genes = list(adata.obs["gene"])
    expression_gene = list(adata.var.index) + ['non-targeting']
    gene_to_interventions_ser = adata[adata.obs['gene'].isin(expression_gene)].obs.reset_index().reset_index().groupby('gene')['index'].apply(list)
    intervened_genes_set = set(gene_to_interventions_ser[gene_to_interventions_ser.apply(len) > k].index)
    intervened_genes = ["excluded" if gene not in intervened_genes_set else gene for gene in intervened_genes]
    # self.adata = self.adata[:,self.adata.var.index.isin(intervened_genes_set)]
    adata.obs['gene'] = intervened_genes
    adata = adata[adata.obs['gene'] != "excluded"]

    dosage_df = dosage_df[sorted(dosage_df.columns)]
    hvg = self.adata.var[self.adata.var['highly_variable']].index
    self.additional_dosage_ls = sorted(set(hvg) - set(dosage_df.columns))
    additional_dosage_df = pd.DataFrame(False, index=dosage_df.index, columns=self.additional_dosage_ls)
    dosage_df = pd.concat([dosage_df, additional_dosage_df], axis=1)




def get_qc_annotation(adata):
    print(f"START : get QC annoation\t{datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')}")
    adata.var.rename(columns={'index':'ensemble_id'}, inplace=True)
    adata.var['ncounts']    = adata.X.sum(axis=0).tolist()[0]
    adata.var['ncells']     = (adata.X > 0).sum(axis=0).tolist()[0]
    adata.obs['UMI_count']  = adata.X.sum(axis=1)
    adata.obs['ngenes']     = (adata.X > 0).sum(axis=1)
    adata.var["mt"]         = adata.var.index.str.startswith("MT-")
    adata.var["ribo"]       = adata.var.index.str.startswith(("RPS", "RPL"))
    adata.var["hb"]         = adata.var.index.str.contains("^HB[^(P)]")
    # with cp.cuda.Device(5):
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True)
    rsc.pp.scrublet(adata)
    rsc.get.anndata_to_CPU(adata)
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print(f"DONE  : get QC annoation\t{datetime.now(timezone('Asia/Seoul')).strftime('%H:%M:%S')}")
    return adata


def get_qc_one_hot_cols(adata, thr, thr_values):
    adata.obs['qc_UMI_count']           = _get_result_of_qc(adata.obs['UMI_count'], metric='mad', thr=-thr, threshold = thr_values['thr_qc_UMI_count'])
    adata.obs['qc_ngenes']              = _get_result_of_qc(adata.obs['ngenes'], metric='mad', thr=-thr, threshold = thr_values['thr_qc_ngenes'])
    adata.obs['qc_pct_counts_mt']       = _get_result_of_qc(adata.obs['pct_counts_mt'], metric='mad', thr=thr, threshold = thr_values['thr_qc_pct_counts_mt'])
    adata.obs['qc_pct_counts_ribo']     = _get_result_of_qc(adata.obs['pct_counts_ribo'], metric='mad', thr=thr, threshold = thr_values['thr_qc_pct_counts_ribo'])
    adata.obs['qc_pct_counts_hb']       = _get_result_of_qc(adata.obs['pct_counts_hb'], metric='mad', thr=thr, threshold = thr_values['thr_qc_pct_counts_hb'])
    adata.obs['qc_predicted_doublet']   = (adata.obs['predicted_doublet'] == True).astype(int)
    qc_one_hot_cols = [col for col in adata.obs.columns if "qc_" in col]
    adata.obs["num_qc"] = adata.obs[qc_one_hot_cols].sum(1)
    adata.obs['total_qc'] = (adata.obs["num_qc"]>0).astype(int)

    return adata, qc_one_hot_cols


def _get_result_of_qc(x_series, metric = 'iqr', thr=1.5, threshold=None):

    if threshold == None:
        if metric == 'iqr':
            Q1 = np.percentile(x_series, 25)
            Q3 = np.percentile(x_series, 75)
            IQR = Q3 - Q1
            threshold = Q3 + thr * IQR # 1.5, 3, 4.5
        elif metric == 'mad': # median absolute deviation
            med = np.median(x_series)
            MAD = np.median(abs(x_series-med))
            threshold = med + MAD * thr
    
    if thr < 0:
        result_of_qc = (x_series < threshold).astype(int)
    elif thr > 0:
        result_of_qc = (x_series > threshold).astype(int)

    return result_of_qc