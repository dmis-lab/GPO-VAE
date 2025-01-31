from typing import Literal, Optional, Sequence

import gc
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import torch
from torch.utils.data import DataLoader

import anndata
import scanpy as sc
import rapids_singlecell as rsc
import cupy as cp

import pertpy as pt
import ot

from gpo_vae.analysis.average_treatment_effects import (
    estimate_data_average_treatment_effects,
)
from gpo_vae.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
    PerturbationDataModule,
)
from gpo_vae.data.utils.perturbation_dataset import SCRNASeqTensorPerturbationDataset


class AdamsonDataModule(PerturbationDataModule):
    def __init__(
        self,
        # deprecated argument
        data_key: Optional[
            Literal["K562_genome_wide_filtered", "K562_essential"]
        ] = None,
        batch_size: int = 128,
        data_path: Optional[str] = None,
        qc_threshold: int = 3,
        n_qc_pass: int = None,
        n_qc_fail: int = None,
        use_each_qc: bool = False,
        stat_path: Optional[str] = None,
        seed: int = 0,
        get_pair_mode: str = 'wd',
        mask_test: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.qc_threshold = qc_threshold

        if data_path is None:
            raise "There is no data"
        
        self.adata = anndata.read_h5ad(data_path)
        if 'qc' not in data_path:
            sub_adata_list = []
            n_each_pert = self.adata.shape[0]/len(self.adata.obs.gene.unique())
            step = int(n_each_pert/10)
            for i in tqdm(range(11)):
                if i == 10:
                    idx = self.adata.obs.groupby('gene').apply(lambda x: x[i*step:]).index.get_level_values(1)
                    sub_adata = self.adata[idx]
                    sub_adata = get_qc_annotation(sub_adata)
                    sub_adata_list.append(sub_adata)
                else:
                    idx = self.adata.obs.groupby('gene').apply(lambda x: x[i*step:(i+1)*step]).index.get_level_values(1)
                    sub_adata = self.adata[idx]
                    sub_adata = get_qc_annotation(sub_adata)
                    sub_adata_list.append(sub_adata)
            self.adata = anndata.concat(sub_adata_list, axis=0)
            
            # self.adata = get_qc_annotation(self.adata)

            ### Causalbench processing ###
            if stat_path is not None:
                summary_stats_adamson = pd.read_csv(stat_path)
                # remove week perturbation gene and week perturbed cell
                strong_perts = get_strong_perts(summary_stats_adamson) + ['non-targeting']
                self.adata = self.adata[self.adata.obs['gene_id'].isin(strong_perts)]
                # why do you run this code?
                self.adata = filter_cells_by_pert_effect(self.adata)
                # # Remove data that don't have the enough data
                self.adata = filter_cells_by_amount(self.adata, k=100)
            ### Causalbench processing ###
                
            self.adata.var.sort_index(inplace=True)
            
            ### annotate pertrubed gene ###
            adata = self.adata.copy()
            adata = get_perturbed_annotation(adata)
            self.adata.var['highly_variable'] = adata.var['highly_variable']
            self.adata.uns = adata.uns
            rank_genes_dict = defaultdict(list)
            for pert in tqdm(self.adata.obs['gene'].unique()):
                if pert == 'non-targeting':
                    continue
                rank_genes = sc.get.rank_genes_groups_df(self.adata, group=pert)
                # rank_genes = rank_genes[(rank_genes['logfoldchanges']<-3) | (rank_genes['logfoldchanges']>3) & (rank_genes['pvals_adj']<0.01)]
                rank_genes = rank_genes[((rank_genes['logfoldchanges'] < -3) | (rank_genes['logfoldchanges'] > 3)) & (rank_genes['pvals_adj'] < 0.01)]
                if len(rank_genes['names']) == 0:
                    # rank_genes_dict[pert].append()
                    continue
                rank_genes_dict[pert].append(rank_genes['names'].tolist()[0])

            rank_gene_set = list(set([i for v in rank_genes_dict.values() for i in v]))
            self.adata.var['FC2'] = False
            self.adata.var.loc[rank_gene_set,'FC2'] = True
            ### annotate pertrubed gene ###
            
            self.adata.write_h5ad(f"{data_path.split('.h5ad')[0]}_qc_hvg_deg.h5ad")


        self.adata, qc_one_hot_cols, self.thr_values = get_qc_one_hot_cols(self.adata, thr=qc_threshold)
        adata = self.adata.copy()
        self.thr_values = defaultdict(dict)
        for qc_thr in [3,4,5]:
            _, _, self.thr_values[qc_thr] = get_qc_one_hot_cols(adata, thr=qc_thr)
        self.n_qc_pass, self.n_qc_fail = self.adata.obs['total_qc'].value_counts()

        qc = self.adata.obs['total_qc'].to_numpy().astype(np.float32)
        self.qc_var_info = pd.DataFrame(index=['total_qc'])
        qc = torch.from_numpy(qc).unsqueeze(1)

        # define splits
        idx = np.arange(self.adata.shape[0])
        train_idx, test_idx = train_test_split(idx, train_size=0.8, random_state=0)
        train_idx, val_idx = train_test_split(train_idx, train_size=0.8, random_state=0)

        self.adata.obs["split"] = None
        self.adata.obs.iloc[train_idx, self.adata.obs.columns.get_loc("split")] = "train"
        self.adata.obs.iloc[val_idx, self.adata.obs.columns.get_loc("split")] = "val"
        self.adata.obs.iloc[test_idx, self.adata.obs.columns.get_loc("split")] = "test"

        # encode dosages
        # combine non-targeting guides to single label
        self.adata.obs["T"] = self.adata.obs["gene"].apply(lambda x: "non-targeting" if "non-targeting" in x else x)
        dosage_df = pd.get_dummies(self.adata.obs["T"])
        dosage_df = dosage_df.drop(columns=["non-targeting"])
        dosage_df = self._get_additional_dosage(dosage_df, criteria='FC2') # FC2.5
        self.d_var_info = dosage_df.T[[]]
        D = torch.from_numpy(dosage_df.to_numpy().astype(np.float32))
        if mask_test:
            mask_ls = self.d_var_info.index[[579,449,454]].tolist() # out large
            D_mask = [self.d_var_info.index.tolist().index(p) for p in mask_ls]
            D[:,D_mask] = 0

        """
        dosage_df = pd.get_dummies(self.adata.obs["T"])
        # encode non-targeting guides as 0
        dosage_df = dosage_df.drop(columns=["non-targeting"])

        self.d_var_info = dosage_df.T[[]]
        D = torch.from_numpy(dosage_df.to_numpy().astype(np.float32))
        """

        X = torch.from_numpy(self.adata.X)

        ids_tr = self.adata.obs[self.adata.obs["split"] == "train"].index
        X_tr = X[(self.adata.obs["split"] == "train").to_numpy()]
        D_tr = D[(self.adata.obs["split"] == "train").to_numpy()]

        ids_val = self.adata.obs[self.adata.obs["split"] == "val"].index
        X_val = X[(self.adata.obs["split"] == "val").to_numpy()]
        D_val = D[(self.adata.obs["split"] == "val").to_numpy()]

        ids_test = self.adata.obs[self.adata.obs["split"] == "test"].index
        X_test = X[(self.adata.obs["split"] == "test").to_numpy()]
        D_test = D[(self.adata.obs["split"] == "test").to_numpy()]

        qc_tr = qc[(self.adata.obs["split"] == "train").to_numpy()]
        qc_val = qc[(self.adata.obs["split"] == "val").to_numpy()]
        qc_test = qc[(self.adata.obs["split"] == "test").to_numpy()]

        anno = f"matched_ctrl_idx_{seed}"
        self.adata = get_ctrl_pair_annotation(adata_origin=self.adata, anno=anno, mode=get_pair_mode) # ot, wd
        deltaX_tr, deltaX_val, deltaX_test = self._get_deltaX(adata_origin=self.adata, d_var_info=self.d_var_info, seed=seed, anno=anno, mode=get_pair_mode)

        cfX_tr = self._get_cfX(adata=self.adata, X=X, split="train")
        torch.cuda.empty_cache()
        cfX_val = self._get_cfX(adata=self.adata, X=X, split="val")
        torch.cuda.empty_cache()
        cfX_test = self._get_cfX(adata=self.adata, X=X, split="test")
        torch.cuda.empty_cache()
        
        self.train_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_tr, D=D_tr, ids=ids_tr, qc=qc_tr, cfX=cfX_tr, deltaX = deltaX_tr,
        )
        self.val_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_val, D=D_val, ids=ids_val, qc=qc_val, cfX=cfX_val, deltaX = deltaX_val,
        )
        self.test_dataset = SCRNASeqTensorPerturbationDataset(
            X=X_test, D=D_test, ids=ids_test, qc=qc_test, cfX=cfX_test, deltaX = deltaX_test,
        )

        x_tr_mean = X_tr.mean(0)
        x_tr_std = X_tr.std(0)
        log_x_tr = torch.log(X_tr + 1)
        log_x_tr_mean = log_x_tr.mean(0)
        log_x_tr_std = log_x_tr.std(0)

        self.x_train_statistics = ObservationNormalizationStatistics(
            x_mean=x_tr_mean,
            x_std=x_tr_std,
            log_x_mean=log_x_tr_mean,
            log_x_std=log_x_tr_std,
        )

        # because there are no perturbation combinations in this simulation,
        # unique_perturbations are the same as the observed perturbations
        # generate unique intervention info dataframe
        df = self.adata.obs.groupby("T")["split"].agg(set).reset_index()
        for split in ["train", "val", "test"]:
            df[split] = df["split"].apply(lambda x: split in x)
        df = df.set_index("T").drop(columns=["split"])
        self.unique_observed_intervention_df = df

        # generate mapping from intervention names to dosages
        self.adata.obs["i"] = np.arange(self.adata.shape[0])
        idx_map = self.adata.obs.drop_duplicates("T").set_index("T")["i"].to_dict()
        self.unique_intervention_dosage_map = {k: D[v] for k, v in idx_map.items()}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_perturbation_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_dosage_obs_per_dim()

    def get_val_perturbation_obs_counts(self) -> torch.Tensor:
        #
        return self.val_dataset.get_dosage_obs_per_dim()

    def get_test_perturbation_obs_counts(self) -> torch.Tensor:
        return self.test_dataset.get_dosage_obs_per_dim()

    def get_train_qc_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_qc_obs_per_dim()

    def get_val_qc_obs_counts(self) -> torch.Tensor:
        return self.val_dataset.get_qc_obs_per_dim()

    def get_test_qc_obs_counts(self) -> torch.Tensor:
        return self.test_dataset.get_qc_obs_per_dim()

    def get_x_var_info(self) -> pd.DataFrame:
        return self.adata.var.copy()

    def get_d_var_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_qc_var_info(self) -> pd.DataFrame:
        return self.qc_var_info.copy()
    
    def get_qc_n(self) -> dict:
        return {'n_qc_pass':self.n_qc_pass,
                'n_qc_fail':self.n_qc_fail}

    def get_obs_info(self) -> pd.DataFrame:
        return self.adata.obs.copy()

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        return self.x_train_statistics

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        return self.unique_observed_intervention_df.copy()

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> torch.Tensor:
        D = torch.zeros((len(pert_names), self.d_var_info.shape[0]))
        for i, pert_name in enumerate(pert_names):
            D[i] = self.unique_intervention_dosage_map[pert_name]
        return D

    def get_estimated_average_treatment_effects(
        self,
        method: Literal["mean", "perturbseq"],
        split: Optional[str] = None,
        qc_pass: bool = False,
    ) -> Optional[anndata.AnnData]:
        adata = self.adata
        if qc_pass:
            adata = adata[adata.obs['total_qc']==0]
        if split is not None:
            adata = adata[adata.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata,
            label_col="T",
            control_label="non-targeting",
            method=method,
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return None
    
    def _get_cfX(self, adata, X, split):        
        print("get_cfX\tSTART")
        idx_per_t_qc_pass = adata[(adata.obs["split"] == split)&(adata.obs["total_qc"] == 0)].obs.groupby("T", observed=False).indices
        idx_per_t_qc_fail = adata[(adata.obs["split"] == split)&(adata.obs["total_qc"] == 1)].obs.groupby("T", observed=False).indices
        pass_fail = set(idx_per_t_qc_pass.keys()) - set(idx_per_t_qc_fail.keys())
        fail_pass = set(idx_per_t_qc_fail.keys()) - set(idx_per_t_qc_pass.keys())
        for i in pass_fail:
            idx_per_t_qc_fail[i] = np.array([])
        for i in fail_pass:
            idx_per_t_qc_pass[i] = np.array([])
        # cf_idx = adata[adata.obs["split"] == split].obs.apply(lambda x: idx_per_t_qc_pass[x.treatment] if x.total_qc == 1 else idx_per_t_qc_fail[x.treatment], axis=1)
        cf_idx = adata[adata.obs["split"] == split].obs.apply(lambda x: np.random.choice(idx_per_t_qc_fail[x["T"]], min(len(idx_per_t_qc_fail[x["T"]]), 50)), axis=1)
        X = X.cuda()
        # cfX = torch.stack(cf_idx.apply(lambda i: X[i].mean(0).detach().cpu() if len(i)>0 else torch.zeros(X.shape[1])).to_list())
        cfX = torch.stack(cf_idx.apply(lambda i: X[i].median(0)[0].detach().cpu() if len(i)>0 else torch.zeros(X.shape[1])).to_list())
        print("get_cfX\tDone")
        return cfX
    
    def _get_additional_dosage(self, dosage_df, criteria):
        dosage_df = dosage_df[sorted(dosage_df.columns)]
        hvg = self.adata.var[self.adata.var[criteria]].index
        self.additional_dosage_ls = sorted(set(hvg) - set(dosage_df.columns))
        additional_dosage_df = pd.DataFrame(False, index=dosage_df.index, columns=self.additional_dosage_ls)
        dosage_df = pd.concat([dosage_df, additional_dosage_df], axis=1)
        return dosage_df
    
    def _get_deltaX(self, adata_origin, d_var_info, seed, anno, mode):
        adata = adata_origin.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata_d = adata[:,d_var_info.index]
        # deltaX = abs(adata_d.X - adata_d.X[adata_d.obs['matched_ctrl_idx_0_not_norm'].astype(int).tolist()])
        deltaX = abs(adata_d.X - adata_d.X[adata_d.obs[f'{anno}_{mode}'].astype(int).tolist()])
        # deltaX = normalize_min_max(deltaX)
        deltaX = np.nan_to_num(deltaX)
        deltaX_tr = deltaX[(adata.obs["split"] == "train").to_numpy()]
        deltaX_val = deltaX[(adata.obs["split"] == "val").to_numpy()]
        deltaX_test = deltaX[(adata.obs["split"] == "test").to_numpy()]
        return deltaX_tr, deltaX_val, deltaX_test


def get_qc_annotation(adata):
    print('START : get QC annoation...')
    if adata.var.index.name != 'gene_name': 
        adata.var = adata.var.rename_axis('gene_name', axis='index')
    adata.var = adata.var.reset_index().set_index('gene_name').rename(columns={'gene_id':'ensemble_id'})
    adata.var.index = adata.var.index.astype(str)
    adata.var_names_make_unique()
    adata.var['ncounts']    = adata.X.sum(axis=0).tolist()[0]
    adata.var['ncells']     = (adata.X > 0).sum(axis=0).tolist()[0]
    adata.obs['UMI_count']  = adata.X.sum(axis=1)
    adata.obs['ngenes']     = (adata.X > 0).sum(axis=1)
    adata.var["mt"]         = adata.var.index.str.startswith("MT-")
    adata.var["ribo"]       = adata.var.index.str.startswith(("RPS", "RPL"))
    adata.var["hb"]         = adata.var.index.str.contains("^HB[^(P)]")

    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], log1p=True)
    rsc.pp.scrublet(adata)
    rsc.get.anndata_to_CPU(adata)

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print('DONE  : get QC annoation...')
    return adata


def get_qc_one_hot_cols(adata, thr):
    thr_values = {}
    adata.obs['qc_UMI_count'], thr_values['thr_qc_UMI_count']            = _get_result_of_qc(adata.obs['UMI_count'], metric='mad', thr=-thr)
    adata.obs['qc_ngenes'], thr_values['thr_qc_ngenes']                  = _get_result_of_qc(adata.obs['ngenes'], metric='mad', thr=-thr)
    adata.obs['qc_pct_counts_mt'], thr_values['thr_qc_pct_counts_mt']    = _get_result_of_qc(adata.obs['pct_counts_mt'], metric='mad', thr=thr)
    adata.obs['qc_pct_counts_ribo'], thr_values['thr_qc_pct_counts_ribo']= _get_result_of_qc(adata.obs['pct_counts_ribo'], metric='mad', thr=thr)
    adata.obs['qc_pct_counts_hb'], thr_values['thr_qc_pct_counts_hb']    = _get_result_of_qc(adata.obs['pct_counts_hb'], metric='mad', thr=thr)
    adata.obs['qc_predicted_doublet']   = (adata.obs['predicted_doublet'] == True).astype(int)
    qc_one_hot_cols = [col for col in adata.obs.columns if "qc_" in col]
    adata.obs["num_qc"] = adata.obs[qc_one_hot_cols].sum(1)
    adata.obs['total_qc'] = (adata.obs["num_qc"]>0).astype(int)

    return adata, qc_one_hot_cols, thr_values


def _get_result_of_qc(x_series, metric = 'iqr', thr=1.5):

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

    return result_of_qc, threshold

def get_strong_perts(supp):
    print('### Preprocessing : Get strong perts ###')

    filtered = supp[supp['Number of DEGs (anderson-darling)']>50]
    filtered = filtered[filtered['percent knockdown']<=-0.3]
    filtered = filtered[filtered['number of cells (filtered)']>25]
    strong_perts = filtered['genetic perturbation'].values
    strong_perts = [s for s in strong_perts]
    return strong_perts


def filter_cells_by_pert_effect(adata, k=10):
    print('### Preprocessing : Filter cells by pert effect ###')

    subset_idxs = []
    ctrl_adata = adata[adata.obs['gene'] == 'non-targeting']

    for itr, pert_gene in enumerate(tqdm(adata.obs['gene'].unique())):        
        subset = adata[adata.obs['gene'] == pert_gene]

        if pert_gene == 'non-targeting':
            subset_idxs.append(subset.obs.index.values)
            continue

        if pert_gene in adata.var.index:
            thresh = np.percentile(ctrl_adata[:,pert_gene].X,k)
            filtered_subset = subset[subset[:,pert_gene].X<=thresh]
            subset_idxs.append(filtered_subset.obs.index.values)
        else:
            subset_idxs.append(subset.obs.index.values)

    subset_idxs = [item for sublist in subset_idxs for item in sublist]
    filtered_adata = adata[subset_idxs,:]

    return filtered_adata


def filter_cells_by_amount(adata, k=100):
    print('### Preprocessing : Filter cells by amount of pert data ###')

    intervened_genes = list(adata.obs["gene"])
    expression_gene = list(adata.var.index) + ['non-targeting']
    gene_to_interventions_ser = adata[adata.obs['gene'].isin(expression_gene)].obs.reset_index().reset_index().groupby('gene')['index'].apply(list)
    intervened_genes_set = set(gene_to_interventions_ser[gene_to_interventions_ser.apply(len) > k].index)
    intervened_genes = ["excluded" if gene not in intervened_genes_set else gene for gene in intervened_genes]
    # self.adata = self.adata[:,self.adata.var.index.isin(intervened_genes_set)]
    adata.obs['gene'] = intervened_genes
    adata = adata[adata.obs['gene'] != "excluded"]

    return adata


def get_perturbed_annotation(adata):
    print('### START : get perturbed gene annoatation... ###')

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, batch_key="gene")
    sc.tl.rank_genes_groups(adata, groupby="gene", method="wilcoxon", reference="non-targeting")

    print('### DONE  : get perturbed gene annoatation... ###')
    return adata


def normalize_min_max(matrix):
    min_vals = matrix.min(axis=0, keepdims=True)
    max_vals = matrix.max(axis=0, keepdims=True)
    range_vals = np.maximum(max_vals - min_vals, 1e-8)
    return (matrix - min_vals) / range_vals

def calculate_wd(control_X, perturbation_X):
    wd_np = np.array([wasserstein_distance(control_X[:, i], perturbation_X[:, i])for i in range(control_X.shape[1])])
    return [np.matmul(abs(pert - control_X), wd_np).argmax() for pert in perturbation_X], wd_np
    # return [np.matmul(normalize_min_max(abs(pert - control_X)), normalize_min_max(wd_np)).argmax() for pert in perturbation_X], wd_np

def calculate_ot(control_X, perturbation_X):
    p = np.ones(perturbation_X.shape[0]) / perturbation_X.shape[0]
    q = np.ones(control_X.shape[0]) / control_X.shape[0]
    cost_matrix = ot.dist(perturbation_X, control_X, metric='euclidean') / 100.0
    gamma = ot.sinkhorn(p, q, cost_matrix, 1.0)
    return gamma.argmax(axis=1).tolist(), None

def hungarian_partial_match(control_X, perturbation_X):
    n1 = perturbation_X.shape[0]
    n2 = control_X.shape[0]

    cost = np.linalg.norm(perturbation_X[:, None, :] - control_X[None, :, :],axis=2)
    s = max(n1, n2)
    big_cost = np.full((s, s), fill_value=1e9)
    big_cost[:n1, :n2] = cost

    row_ind, col_ind = linear_sum_assignment(big_cost)
    match_indices = [c for r,c in zip(row_ind, col_ind) if r<n1 and c<n2]
    return match_indices, None

def process_gene(g, perturbation_X, control_X, mode):
    if mode == "wd":
        return calculate_wd(control_X, perturbation_X)
    elif mode == "ot":
        return calculate_ot(control_X, perturbation_X)
    elif mode == "hpm":
        return hungarian_partial_match(control_X, perturbation_X)

def process_data(data, control_X, mode):
    g, perturbation_X = data
    return process_gene(g, perturbation_X, control_X, mode)

def get_ctrl_pair_annotation(adata_origin, anno="matched_ctrl_idx", mode="ot"):
    print('### START : get ctrl pair annotation... ###')

    if f'{anno}_{mode}' in adata_origin.obs.columns: #  or f'matched_ctrl_idx_0_not_norm' in adata_origin.obs.columns
        print('### DONE : Already it had a ctrl pair!!! ###')
        return adata_origin

    adata_origin.obs[anno] = '-1'
    adata = adata_origin.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    pert_gene_ls = sorted([g for g in adata.obs.gene.unique() if g != 'non-targeting'])
    adata = adata[:, pert_gene_ls]

    for split_mode in ["train", "val", "test"]:
        adata_split = adata[adata.obs['split'] == split_mode]
        control_adata = adata_split[adata_split.obs['gene'] == 'non-targeting']
        control_X = control_adata.X.toarray()
        ctrl_idx_to_origidx_dict = dict(enumerate(control_adata.obs.index)) # control

        genes = pert_gene_ls + ['non-targeting']
        perturbation_data = [(g, adata_split[adata_split.obs['gene'] == g].X.toarray()) for g in genes]

        
        with tqdm(total=len(genes), desc=f"Processing {split_mode}") as pbar:
            results = Parallel(n_jobs=20)(
                delayed(process_data)(data, control_X, mode) for data in perturbation_data
            )

        global_control_idx = [idx for sublist in results for idx in sublist[0]]
        adata_origin.obs.loc[adata_origin.obs['split'] == split_mode, f'{anno}_{mode}'] = [ctrl_idx_to_origidx_dict[i] for i in global_control_idx]
        if mode == 'wd':
            adata_origin.uns[f'{anno}_{mode}_{split_mode}'] = np.array([sublist[1] for sublist in results])[:-1,:]

    adata_origin.write_h5ad(f"./datasets/adamson_qc_deg_{anno}_{mode}.h5ad")
    print('### DONE  : get ctrl pair annotation... ###')
    return adata_origin