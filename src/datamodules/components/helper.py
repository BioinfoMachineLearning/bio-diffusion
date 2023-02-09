# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
from typing import Union

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@typechecked
def _normalize(
    tensor: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    From https://github.com/drorlab/gvp-pytorch
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )

@typechecked
def _rbf(
    D: torch.Tensor,
    D_min: float = 0.0,
    D_max: float = 20.0,
    D_count: int = 16,
    device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
