# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch

import numpy as np

from copy import copy
from functools import partial
from typing import Any, Optional, Tuple, Union
from torch import nn

from torch_geometric.data import Batch
from torch_scatter import scatter

from omegaconf import OmegaConf, DictConfig

from src.datamodules.components.edm_dataset import _edge_features, _node_features

from src.models import get_nonlinearity
from src.models.components import GCPDropout, GCPLayerNorm, ScalarVector, centralize, is_identity, localize, safe_norm, scalarize, vectorize
from src.models.components.variational_diffusion import NODE_FEATURE_DIFFUSION_TARGETS
from src.utils.pylogger import get_pylogger

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


class GCP(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
            scalar_out_nonlinearity: Optional[str] = "silu",
            scalar_gate: int = 0,
            vector_gate: bool = True,
            frame_gate: bool = False,
            sigma_frame_gate: bool = False,
            feedforward_out: bool = False,
            bottleneck: int = 1,
            vector_residual: bool = False,
            vector_frame_residual: bool = False,
            ablate_frame_updates: bool = False,
            ablate_scalars: bool = False,
            ablate_vectors: bool = False,
            scalarization_vectorization_output_dim: int = 3,
            **kwargs
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = (None, None)

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True)
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate, vector_gate, frame_gate, sigma_frame_gate
        )
        self.vector_residual, self.vector_frame_residual = vector_residual, vector_frame_residual
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Sequential(
                nn.Linear(self.hidden_dim + self.scalar_input_dim, self.scalar_output_dim),
                get_nonlinearity(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.hidden_dim + self.scalar_input_dim, self.scalar_output_dim)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)

            if not self.ablate_frame_updates:
                vector_down_frames_input_dim = self.hidden_dim if not self.vector_output_dim else self.vector_output_dim
                self.vector_down_frames = nn.Linear(vector_down_frames_input_dim,
                                                    scalarization_vectorization_output_dim, bias=False)
                self.scalar_out_frames = nn.Linear(
                    self.scalar_output_dim + scalarization_vectorization_output_dim * 3, self.scalar_output_dim)

                if self.vector_output_dim and self.sigma_frame_gate:
                    self.vector_out_scale_sigma_frames = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
                elif self.vector_output_dim and self.frame_gate:
                    self.vector_out_scale_frames = nn.Linear(
                        self.scalar_output_dim, scalarization_vectorization_output_dim * 3)
                    self.vector_up_frames = nn.Linear(
                        scalarization_vectorization_output_dim, self.vector_output_dim, bias=False)
        else:
            self.scalar_out = nn.Sequential(
                nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                get_nonlinearity(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    @typechecked
    def process_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)
        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def create_zero_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)

    @typechecked
    def process_vector_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "o"],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_entities", "p", 3]:
        vector_rep = v_pre.transpose(-1, -2)
        if self.sigma_frame_gate:
            # bypass vectorization in favor of row-wise gating
            gate = self.vector_out_scale_sigma_frames(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif self.frame_gate:
            # apply elementwise gating between localized frame vectors and vector residuals
            gate = self.vector_out_scale_frames(self.vector_nonlinearity(scalar_rep))
            # perform frame-gating, where edges must be present
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask
            )
            # ensure the channels for `coordinates` are being left-multiplied
            gate_vector_rep = self.vector_up_frames(gate_vector.transpose(-1, -2)).transpose(-1, -2)
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(gate_vector_rep, dim=-1, keepdim=True))
            if self.vector_frame_residual:
                vector_rep = vector_rep + v_pre.transpose(-1, -2)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                TensorType["batch_num_entities", "scalar_dim"],
                TensorType["batch_num_entities", "m", "vector_dim"]
            ],
            TensorType["batch_num_entities", "merged_scalar_dim"]
        ],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool = False,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_entities", "new_scalar_dim"],
            TensorType["batch_num_entities", "n", "vector_dim"]
        ],
        TensorType["batch_num_entities", "new_scalar_dim"]
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)
        else:
            merged = s_maybe_v
            merged = torch.zeros_like(merged) if self.ablate_scalars else merged

        scalar_rep = self.scalar_out(merged)

        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector(scalar_rep, v_pre, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        vector_rep = self.create_zero_vector(
            scalar_rep
        ) if self.vector_output_dim and not self.vector_input_dim else vector_rep

        if self.ablate_frame_updates:
            return ScalarVector(scalar_rep, vector_rep) if self.vector_output_dim else scalar_rep

        # GCP: update scalar features using complete local frames
        v_pre = vector_rep.transpose(-1, -2)
        vector_hidden_rep = self.vector_down_frames(v_pre)
        scalar_hidden_rep = scalarize(
            vector_hidden_rep.transpose(-1, -2),
            edge_index,
            frames,
            node_inputs=node_inputs,
            dim_size=vector_hidden_rep.shape[0],
            node_mask=node_mask
        )
        merged = torch.cat((scalar_rep, scalar_hidden_rep), dim=-1)

        scalar_rep = self.scalar_out_frames(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using complete local frames (e.g., in the case of a final layer)
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            return self.scalar_nonlinearity(scalar_rep)

        # GCP: update vector features using complete local frames
        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector_frames(
                scalar_rep,
                v_pre,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        return ScalarVector(scalar_rep, vector_rep)


class GCP2(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
            scalar_out_nonlinearity: Optional[str] = "silu",
            scalar_gate: int = 0,
            vector_gate: bool = True,
            frame_gate: bool = False,
            sigma_frame_gate: bool = False,
            feedforward_out: bool = False,
            bottleneck: int = 1,
            vector_residual: bool = False,
            vector_frame_residual: bool = False,
            ablate_frame_updates: bool = False,
            ablate_scalars: bool = False,
            ablate_vectors: bool = False,
            scalarization_vectorization_output_dim: int = 3,
            **kwargs
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = (None, None)

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True)
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate, vector_gate, frame_gate, sigma_frame_gate
        )
        self.vector_residual, self.vector_frame_residual = vector_residual, vector_frame_residual
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            scalar_vector_frame_dim = (
                (scalarization_vectorization_output_dim * 3)
                if not self.ablate_frame_updates
                else 0
            )
            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Sequential(
                nn.Linear(self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim, self.scalar_output_dim),
                get_nonlinearity(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim, self.scalar_output_dim)

            if not self.ablate_frame_updates:
                self.vector_down_frames = nn.Linear(
                    self.vector_input_dim, scalarization_vectorization_output_dim, bias=False)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if not self.ablate_frame_updates:
                    if self.frame_gate:
                        self.vector_out_scale_frames = nn.Linear(
                            self.scalar_output_dim, scalarization_vectorization_output_dim * 3)
                        self.vector_up_frames = nn.Linear(
                            scalarization_vectorization_output_dim, self.vector_output_dim, bias=False)
                    elif self.vector_gate:
                        self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
                elif self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
        else:
            self.scalar_out = nn.Sequential(
                nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                get_nonlinearity(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    @typechecked
    def create_zero_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)

    @typechecked
    def process_vector_without_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def process_vector_with_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)

        if self.frame_gate:
            # derive vector features from direction-robust frames
            gate = self.vector_out_scale_frames(self.vector_nonlinearity(scalar_rep))
            # perform frame-gating, where edges must be present
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask
            )
            # ensure frame vector channels for `coordinates` are being left-multiplied
            gate_vector_rep = self.vector_up_frames(gate_vector.transpose(-1, -2)).transpose(-1, -2)
            # apply row-wise scalar gating with frame vector
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(gate_vector_rep, dim=-1, keepdim=True))
        elif self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                TensorType["batch_num_entities", "scalar_dim"],
                TensorType["batch_num_entities", "m", "vector_dim"]
            ],
            TensorType["batch_num_entities", "merged_scalar_dim"]
        ],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool = False,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_entities", "new_scalar_dim"],
            TensorType["batch_num_entities", "n", "vector_dim"]
        ],
        TensorType["batch_num_entities", "new_scalar_dim"]
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            if not self.ablate_frame_updates:
                # GCP2: curate direction-robust scalar geometric features
                vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
                scalar_hidden_rep = scalarize(
                    vector_down_frames_hidden_rep.transpose(-1, -2),
                    edge_index,
                    frames,
                    node_inputs=node_inputs,
                    dim_size=vector_down_frames_hidden_rep.shape[0],
                    node_mask=node_mask
                )
                merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        elif self.ablate_frame_updates:
            # GCP-Baseline: update vector features using row-wise scalar gating
            vector_rep = self.process_vector_without_frames(scalar_rep, v_pre, vector_hidden_rep)
        else:
            # GCP2: update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.process_vector_with_frames(
                scalar_rep,
                v_pre,
                vector_hidden_rep,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        return ScalarVector(scalar_rep, vector_rep)


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: ScalarVector,
        node_input_dims: ScalarVector,
        edge_hidden_dims: ScalarVector,
        node_hidden_dims: ScalarVector,
        num_atom_types: int,
        nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
        cfg: DictConfig = None,
        pre_norm: bool = True,
        use_gcp_norm: bool = True
    ):
        super().__init__()

        if num_atom_types > 0:
            self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        else:
            self.atom_embedding = None

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(edge_input_dims, use_gcp_norm=use_gcp_norm)
            self.node_normalization = GCPLayerNorm(node_input_dims, use_gcp_norm=use_gcp_norm)
        else:
            self.edge_normalization = GCPLayerNorm(edge_hidden_dims, use_gcp_norm=use_gcp_norm)
            self.node_normalization = GCPLayerNorm(node_hidden_dims, use_gcp_norm=use_gcp_norm)

        self.edge_embedding = cfg.selected_GCP(
            edge_input_dims,
            edge_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            frame_gate=cfg.frame_gate,
            sigma_frame_gate=cfg.sigma_frame_gate,
            vector_frame_residual=cfg.vector_frame_residual,
            ablate_frame_updates=cfg.ablate_frame_updates,
            ablate_scalars=cfg.ablate_scalars,
            ablate_vectors=cfg.ablate_vectors
        )

        self.node_embedding = cfg.selected_GCP(
            node_input_dims,
            node_hidden_dims,
            nonlinearities=(None, None),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            frame_gate=cfg.frame_gate,
            sigma_frame_gate=cfg.sigma_frame_gate,
            vector_frame_residual=cfg.vector_frame_residual,
            ablate_frame_updates=cfg.ablate_frame_updates,
            ablate_scalars=cfg.ablate_scalars,
            ablate_vectors=cfg.ablate_vectors
        )

    @typechecked
    def forward(
        self,
        batch: Batch
    ) -> Tuple[
        Union[
            Tuple[
                TensorType["batch_num_nodes", "h_hidden_dim"],
                TensorType["batch_num_nodes", "m", "chi_hidden_dim"]
            ],
            TensorType["batch_num_nodes", "h_hidden_dim"]
        ],
        Union[
            Tuple[
                TensorType["batch_num_edges", "e_hidden_dim"],
                TensorType["batch_num_edges", "x", "xi_hidden_dim"]
            ],
            TensorType["batch_num_edges", "e_hidden_dim"]
        ]
    ]:
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)

        edge_rep = edge_rep.scalar if not self.edge_embedding.vector_input_dim else edge_rep
        node_rep = node_rep.scalar if not self.node_embedding.vector_input_dim else node_rep

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None)
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None)
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep


def get_GCP_with_custom_cfg(input_dims, output_dims, cfg: DictConfig, **kwargs):
    cfg_dict = copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    cfg_dict["nonlinearities"] = cfg.nonlinearities
    del cfg_dict["scalar_nonlinearity"]
    del cfg_dict["vector_nonlinearity"]

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return cfg.selected_GCP(input_dims, output_dims, **cfg_dict)


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        mp_cfg: DictConfig,
        reduce_function: str = "sum",
        use_scalar_message_attention: bool = True
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.reduce_function = reduce_function
        self.use_residual_message_gcp = self.conv_cfg.use_residual_message_gcp
        self.use_scalar_message_attention = use_scalar_message_attention

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        # config instantiations
        soft_cfg = copy(cfg)
        soft_cfg.bottleneck, soft_cfg.vector_residual = cfg.default_bottleneck, cfg.default_vector_residual

        primary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=soft_cfg)
        secondary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=cfg)

        # PyTorch modules #
        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=cfg.nonlinearities
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(secondary_cfg_GCP(output_dims, output_dims))

        if self.conv_cfg.num_message_layers > 1:
            module_list.append(primary_cfg_GCP(output_dims, output_dims, nonlinearities=cfg.nonlinearities))

        self.message_fusion = nn.ModuleList(module_list)

        # learnable scalar message gating
        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dims.scalar, 1),
                nn.Sigmoid()
            )

    @typechecked
    def message(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_edges", "message_dim"]:
        row, col = edge_index
        vector = node_rep.vector.reshape(node_rep.vector.shape[0], node_rep.vector.shape[1] * node_rep.vector.shape[2])
        vector_reshaped = ScalarVector(node_rep.scalar, vector)

        s_row, v_row = vector_reshaped.idx(row)
        s_col, v_col = vector_reshaped.idx(col)

        v_row = v_row.reshape(v_row.shape[0], v_row.shape[1] // 3, 3)
        v_col = v_col.reshape(v_col.shape[0], v_col.shape[1] // 3, 3)

        message = ScalarVector(s_row, v_row).concat((edge_rep, ScalarVector(s_col, v_col)))
        
        if self.use_residual_message_gcp:
            message_residual = self.message_fusion[0](message, edge_index, frames, node_inputs=False, node_mask=node_mask)
            for module in self.message_fusion[1:]:
                # ResGCP: exchange geometric messages while maintaining residual connection to original message
                new_message = module(message_residual, edge_index, frames, node_inputs=False, node_mask=node_mask)
                message_residual = message_residual + new_message
        else:
            message_residual = message
            for module in self.message_fusion:
                # ablate ResGCP: exchange geometric messages without maintaining residual connection to original message
                message_residual = module(message_residual, edge_index, frames, node_inputs=False, node_mask=node_mask)

        # learn to gate scalar messages
        if self.use_scalar_message_attention:
            message_residual_attn = self.scalar_message_attention(message_residual.scalar)
            message_residual = ScalarVector(message_residual.scalar * message_residual_attn, message_residual.vector)

        return message_residual.flatten()

    @typechecked
    def aggregate(
        self,
        message: TensorType["batch_num_edges", "message_dim"],
        edge_index: TensorType[2, "batch_num_edges"],
        dim_size: int
    ) -> TensorType["batch_num_nodes", "aggregate_dim"]:
        row, col = edge_index
        aggregate = scatter(message, row, dim=0, dim_size=dim_size, reduce=self.reduce_function)
        return aggregate

    @typechecked
    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> ScalarVector:
        message = self.message(node_rep, edge_rep, edge_index, frames, node_mask=node_mask)
        aggregate = self.aggregate(message, edge_index, dim_size=node_rep.scalar.shape[0])
        return ScalarVector.recover(aggregate, self.vector_output_dim)


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        layer_cfg: DictConfig,
        dropout: float = 0.0,
        nonlinearities: Optional[Tuple[Any, Any]] = None,
        update_node_positions: bool = False
    ):
        super().__init__()

        # hyperparameters #
        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.update_node_positions = update_node_positions
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        self.update_positions_with_vector_sum = getattr(cfg, "update_positions_with_vector_sum", False)
        reduce_function = "sum"

        # PyTorch modules #

        # geometry-complete message-passing neural network
        message_function = GCPMessagePassing

        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function=reduce_function,
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention
        )

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_without_res_cfg = copy(cfg)
        ff_without_res_cfg.vector_residual = False

        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)
        ff_without_res_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_without_res_cfg)

        self.gcp_norm = nn.ModuleList([GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm)])
        self.gcp_dropout = nn.ModuleList([GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout)])

        # build out feedforward (FF) network modules
        hidden_dims = (
            (node_dims.scalar, node_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims.scalar, 2 * node_dims.vector)
        )
        ff_interaction_layers = [
            ff_without_res_GCP(
                (node_dims.scalar * 2, node_dims.vector * 2), hidden_dims,
                nonlinearities=(None, None) if layer_cfg.num_feedforward_layers == 1 else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1
            )
        ]

        interaction_layers = [
            ff_GCP(hidden_dims, hidden_dims)
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_without_res_GCP(
                    hidden_dims, node_dims,
                    nonlinearities=(None, None),
                    feedforward_out=True
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # potentially build out node position update modules
        if update_node_positions:
            # node position update GCP(s)
            position_output_dims = (
                node_dims
                if getattr(cfg, "update_positions_with_vector_sum", False)
                else (node_dims.scalar, 1)
            )
            self.node_position_update_gcp = ff_without_res_GCP(
                node_dims, position_output_dims,
                nonlinearities=cfg.nonlinearities
            )

    @typechecked
    def derive_x_update(
        self,
        node_rep: ScalarVector,
        edge_index: TensorType[2, "batch_num_edges"],
        f_ij: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_nodes", 3]:
        # VectorUpdate: use vector-valued features to derive node position updates
        node_rep_update = self.node_position_update_gcp(
            node_rep,
            edge_index,
            f_ij,
            node_inputs=True,
            node_mask=node_mask
        )
        if self.update_positions_with_vector_sum:
            x_vector_update = node_rep_update.vector.sum(1)
        else:
            x_vector_update = node_rep_update.vector.squeeze(1)

        # (up/down)weight position updates
        x_update = x_vector_update * self.node_positions_weight

        return x_update

    @typechecked
    def forward(
        self,
        node_rep: Tuple[TensorType["batch_num_nodes", "node_hidden_dim"], TensorType["batch_num_nodes", "m", 3]],
        edge_rep: Tuple[TensorType["batch_num_edges", "edge_hidden_dim"], TensorType["batch_num_edges", "x", 3]],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        node_pos: Optional[TensorType["batch_num_nodes", 3]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_nodes", "hidden_dim"],
            TensorType["batch_num_nodes", "n", 3]
        ],
        Tuple[
            Tuple[
                TensorType["batch_num_nodes", "hidden_dim"],
                TensorType["batch_num_nodes", "n", 3]
            ],
            TensorType["batch_num_nodes", 3]
        ]
    ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # forward propagate with interaction module
        hidden_residual = self.interaction(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

        # aggregate input and hidden features
        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep,)))

        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames,
                node_inputs=True,
                node_mask=node_mask
            )

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask.float())

        # bypass updating node positions
        if not self.update_node_positions:
            return node_rep

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        return node_rep, node_pos


class GCPNetDynamics(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        module_cfg: DictConfig,
        layer_cfg: DictConfig,
        diffusion_cfg: DictConfig,
        dataloader_cfg: DictConfig
    ):
        super().__init__()

        # hyperparameters #

        # feature dimensionalities
        h_input_dim_ = (
            dataloader_cfg.num_atom_types + dataloader_cfg.include_charges
            if diffusion_cfg.diffusion_target in NODE_FEATURE_DIFFUSION_TARGETS
            else model_cfg.h_input_dim
        )
        h_input_conditioning_dim = int(diffusion_cfg.condition_on_time)  # time-conditioning
        h_input_conditioning_dim += len(module_cfg.conditioning)  # context-conditioning

        # input feature dimensionalities while considering self-conditioning
        h_input_dim = (
            h_input_dim_ * 2
            if diffusion_cfg.self_condition and diffusion_cfg.diffusion_target in NODE_FEATURE_DIFFUSION_TARGETS
            else h_input_dim_
        )
        e_input_dim = (
            model_cfg.e_input_dim * 2
            if diffusion_cfg.self_condition
            else model_cfg.e_input_dim
        )
        chi_input_dim = (
            model_cfg.chi_input_dim * 2
            if diffusion_cfg.self_condition
            else model_cfg.chi_input_dim
        )
        xi_input_dim = (
            model_cfg.xi_input_dim * 2
            if diffusion_cfg.self_condition
            else model_cfg.xi_input_dim
        )
        self.edge_input_dims = ScalarVector(e_input_dim, xi_input_dim)
        self.node_input_dims = ScalarVector(h_input_dim + h_input_conditioning_dim, chi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)
        self.num_context_node_features = len(module_cfg.conditioning)

        # graph settings
        self.num_x_dims = dataloader_cfg.num_x_dims
        self.norm_x_diff = module_cfg.norm_x_diff
        self.num_radials = dataloader_cfg.num_radials

        # input conditioning settings
        self.self_condition = diffusion_cfg.self_condition
        self.condition_on_time = diffusion_cfg.condition_on_time
        self.condition_on_context = self.num_context_node_features > 0

        # Forward pass #

        forward_prefix_mapping = {
            "atom_types_and_coords": "atom_types_and_coords_"
        }
        forward_prefix = forward_prefix_mapping[diffusion_cfg.diffusion_target]
        self.forward_fn = getattr(self, forward_prefix + "forward")

        # PyTorch modules #

        # input embeddings
        self.gcp_embedding = GCPEmbedding(
            self.edge_input_dims,
            self.node_input_dims,
            self.edge_dims,
            self.node_dims,
            num_atom_types=0,  # note: assumes e.g., input atom types are float values
            cfg=module_cfg,
            use_gcp_norm=layer_cfg.use_gcp_norm
        )

        # message-passing (and node position update) layers
        self.interaction_layers = nn.ModuleList(
            GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
                update_node_positions=True
            ) for _ in range(model_cfg.num_encoder_layers)
        )

        if diffusion_cfg.diffusion_target in NODE_FEATURE_DIFFUSION_TARGETS:
            # scalar node projection via GCP
            h_input_dim_without_self_conditioning = h_input_dim_ + h_input_conditioning_dim
            self.scalar_node_projection_gcp = module_cfg.selected_GCP(
                self.node_dims, (h_input_dim_without_self_conditioning, 0),
                nonlinearities=(None, None),
                scalar_gate=module_cfg.scalar_gate,
                vector_gate=module_cfg.vector_gate,
                frame_gate=module_cfg.frame_gate,
                sigma_frame_gate=module_cfg.sigma_frame_gate,
                vector_frame_residual=module_cfg.vector_frame_residual,
                ablate_frame_updates=module_cfg.ablate_frame_updates,
                ablate_scalars=module_cfg.ablate_scalars,
                ablate_vectors=module_cfg.ablate_vectors
            )

    @typechecked
    def forward(
        self,
        batch: Batch,
        xh: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        t: TensorType["batch_num_nodes", 1],
        **kwargs: Any
    ) -> Tuple[
        Batch,
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
    ]:
        return self.forward_fn(batch, xh, t, **kwargs)

    @staticmethod
    @typechecked
    def get_fully_connected_edge_index(
        batch_index: TensorType["batch_num_nodes"],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType[2, "batch_num_edges"]:
        adj = batch_index[:, None] == batch_index[None, :]
        edge_index = torch.stack(torch.where(adj), dim=0)
        if node_mask is not None:
            row, col = edge_index
            edge_mask = node_mask[row] & node_mask[col]
            edge_index = edge_index[:, edge_mask]
        return edge_index

    @typechecked
    def atom_types_and_coords_forward(
        self,
        batch: Batch,
        xh: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        t: TensorType["batch_num_nodes", 1],
        xh_self_cond: Optional[TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]] = None,
        **kwargs: Any
    ) -> Tuple[
        Batch,
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
    ]:
        # organize input features
        xh = xh.clone() * batch.mask.float().unsqueeze(-1)
        x_init = xh[:, :self.num_x_dims].clone()
        h_init = xh[:, self.num_x_dims:].clone()
        x_self_cond = (
            xh_self_cond[:, :self.num_x_dims].clone()
            if xh_self_cond is not None
            else None
        )
        h_self_cond = (
            xh_self_cond[:, self.num_x_dims:].clone()
            if xh_self_cond is not None
            else None
        )

        # build latest graph topology
        batch.edge_index = self.get_fully_connected_edge_index(
            batch_index=batch.batch,
            node_mask=batch.mask
        )

        # install noisy node positions into graph
        batch.x = x_init

        # install noisy node scalar and vector-valued features into graph
        _, batch.chi = _node_features(batch, edm_sampling=True)
        batch.h = h_init

        # install noisy edge scalar and vector-valued features into graph
        batch.e, batch.xi = _edge_features(batch)

        # self-condition the model's prediction
        if self.self_condition:
            x_self_cond_ = (
                x_self_cond
                if x_self_cond is not None
                else torch.zeros_like(x_init)
            )
            h_self_cond = (
                h_self_cond
                if h_self_cond is not None
                else torch.zeros_like(h_init)
            )
            _, x_self_cond_chi = _node_features(
                Batch(
                    h=batch.h,
                    x=x_self_cond_
                ),
                edm_sampling=True
            )
            x_self_cond_e, x_self_cond_xi = _edge_features(
                Batch(
                    x=x_self_cond_,
                    edge_index=batch.edge_index
                )
            )
            batch.h = torch.cat((batch.h, h_self_cond), dim=-1)
            batch.chi = torch.cat((batch.chi, x_self_cond_chi), dim=1)
            batch.e = torch.cat((batch.e, x_self_cond_e), dim=-1)
            batch.xi = torch.cat((batch.xi, x_self_cond_xi), dim=1)

        # condition model's predictions on the current time step
        if self.condition_on_time:
            if np.prod(t.shape) == 1:
                h_time = torch.empty_like(
                    batch.h[:, 0:1],
                    device=batch.x.device
                ).fill_(t.item())
            else:
                h_time = t.view(batch.num_nodes, 1)
            batch.h = torch.cat((batch.h, h_time), dim=-1)

        # condition model's predictions on the contextual features provided
        if self.condition_on_context:
            context = batch.props_context.view(batch.num_nodes, self.num_context_node_features)
            batch.h = torch.cat((batch.h, context), dim=-1)

        # begin GCPNet forward pass #

        # centralize node positions to necessarily make them translation-invariant during the diffusion process
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True
        )

        # craft complete local frames corresponding to each edge
        batch.f_ij = localize(
            batch.x,
            batch.edge_index,
            norm_x_diff=self.norm_x_diff,
            node_mask=batch.mask
        )

        # embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi), batch.x = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_mask=batch.mask,
                node_pos=batch.x
            )

        # summarize scalar node features using a GCP projection - note: the bias term might be non-zero
        h = self.scalar_node_projection_gcp(
            ScalarVector(h, chi),
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=batch.mask
        )

        # record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # end GCPNet forward pass #

        vel = (batch.x - x_init) * batch.mask.float().unsqueeze(-1)  # use delta(x) to estimate noise
        h_final = batch.h

        # remove contextual features from output predictions if such features were originally provided
        if self.condition_on_context:
            h_final = h_final[:, :-self.num_context_node_features]
        if self.condition_on_time:
            h_final = h_final[:, :-1]  # note: here, the last dimension represents time

        # detect and nullify any invalid node position predictions
        if vel.isnan().any():
            log.warning(f"Detected NaN in `vel` -> resetting GCPNet `vel` output for time step(s) {t} to zero.")
            vel = torch.zeros_like(vel)

        # project output node positions to a zero center of gravity subspace
        batch.vel = vel
        _, vel = centralize(
            batch,
            key="vel",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True
        )
        del batch.vel

        # assemble outputs
        net_out = torch.cat((vel, h_final), dim=-1)

        return batch, net_out


if __name__ == "__main__":
    _ = GCPNetDynamics()
