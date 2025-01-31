import torch.nn as nn

from gpo_vae.models.utils.predictor import PerturbationPlatedPredictor


class gpo_vae_Predictor(PerturbationPlatedPredictor):
    def __init__(self, model: nn.Module, guide: nn.Module):
        super().__init__(
            model=model,
            guide=guide,
            local_variables=["z_basal"], # , "z_noise"
            perturbation_plated_variables=["E", "mask", "qc_E"],
        )
