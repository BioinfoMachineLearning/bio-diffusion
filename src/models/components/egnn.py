# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch_geometric

import numpy as np

from einops import rearrange

from typing import Any, List, Optional, Tuple
from torch import nn, einsum

from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
from torch_geometric.data import Batch

from omegaconf import DictConfig

from src.datamodules.components.edm_dataset import _edge_features, _node_features

from src.models.components import ScalarVector, centralize
from src.models.components.gcpnet import GCPNetDynamics
from src.models.components.variational_diffusion import NODE_FEATURE_DIFFUSION_TARGETS
from src.utils.pylogger import get_pylogger

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)

# helper functions

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


def exists(val):
    return val is not None


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i, emb_layer in enumerate(layers):
        # the portion corresponding to `to_embedd` part gets dropped
        x = torch.cat([x[:, :stop_concat],
                       emb_layer(to_embedd[:, i])
                       ], dim=-1)
        stop_concat = x.shape[-1]
    return x


# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, "SiLU") else Swish_

# global linear attention


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask=None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64
    ):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, mask=None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask=mask)
        out = self.attn2(x, induced)

        x = out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries


class Attention_Sparse(Attention):
    def __init__(self, **kwargs):
        """ Wraps the attention class to operate with pytorch-geometric inputs. """
        super(Attention_Sparse, self).__init__(**kwargs)

    def sparse_forward(self, x, context, batch=None, batch_uniques=None, mask=None):
        assert batch is not None or batch_uniques is not None, "Batch/(uniques) must be passed for block_sparse_attn"
        if batch_uniques is None:
            batch_uniques = torch.unique(batch, return_counts=True)
        # only one example in batch - do dense - faster
        if batch_uniques[0].shape[0] == 1:
            x, context = map(lambda t: rearrange(t, 'h d -> () h d'), (x, context))
            return self.forward(x, context, mask=None).squeeze()  #  get rid of batch dim
        # multiple examples in batch - do block-sparse by dense loop
        else:
            x_list = []
            aux_count = 0
            for bi, n_idxs in zip(*batch_uniques):
                x_list.append(
                    self.sparse_forward(
                        x[aux_count:aux_count + n_i],
                        context[aux_count:aux_count+n_idxs],
                        batch_uniques=(bi.unsqueeze(-1), n_idxs.unsqueeze(-1))
                    )
                )
            return torch.cat(x_list, dim=0)


class GlobalLinearAttention_Sparse(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64
    ):
        super().__init__()
        self.norm_seq = torch_geometric.nn.norm.LayerNorm(dim)
        self.norm_queries = torch_geometric.nn.norm.LayerNorm(dim)
        self.attn1 = Attention_Sparse(dim, heads, dim_head)
        self.attn2 = Attention_Sparse(dim, heads, dim_head)

        # can't concat pyg norms with torch sequentials
        self.ff_norm = torch_geometric.nn.norm.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, batch=None, batch_uniques=None, mask=None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x, batch=batch), self.norm_queries(queries, batch=batch)

        induced = self.attn1.sparse_forward(queries, x, batch=batch, batch_uniques=batch_uniques, mask=mask)
        out = self.attn2.sparse_forward(x, induced, batch=batch, batch_uniques=batch_uniques)

        x = out + res_x
        queries = induced + res_queries

        x_norm = self.ff_norm(x, batch=batch)
        x = self.ff(x_norm) + x_norm
        return x, queries


# define pytorch-geometric equivalents

class EGNN_Sparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and 
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """

    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=16,
        fourier_features=0,
        soft_edge=0,
        coors_tanh=True,
        norm_feats=False,
        norm_coors=True,
        norm_coors_scale_init=1e-2,
        update_feats=True,
        update_coors=True,
        dropout=0.,
        coor_weights_clamp_value=None,
        aggr="add",
        **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None

        assert not (coors_tanh and not norm_coors), 'coors_tanh must be used with norm_coors'

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1),
                                         nn.Sigmoid()
                                         ) if soft_edge else None

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1.
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        # COORS
        # Tanh layer helps with stability but should only be used in conjuction with norm_coors
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1),
            nn.Tanh() if coors_tanh else nn.Identity()
        ) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None,
                angle_data: List = None,  size: Size = None) -> Tensor:
        """ Inputs: 
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (n_edges, 2)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path) long tensor.
            * size: None
        """
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                               coors=coors, rel_coors=rel_coors,
                                               batch=batch)
        return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                coor_weights_clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min=-clamp_value, max=clamp_value)

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = kwargs["x"] + hidden_out
        else:
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__)


class EGNN_Sparse_Network(nn.Module):
    r"""Sample GNN model architecture that uses the EGNN-Sparse
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * ... : same interpretation as the base layer.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * recalc: int. Recalculate edge feats every `recalc` MPNN layers. 0 for no recalc
        * verbose: bool. verbosity level.
        -----
        Diff with normal layer: one has to do preprocessing before (radius, global token, ...)
    """

    def __init__(self, n_layers, feats_dim,
                 pos_dim=3,
                 edge_attr_dim=0,
                 m_dim=16,
                 fourier_features=0,
                 soft_edge=0,
                 coors_tanh=True,
                 embedding_nums=[],
                 embedding_dims=[],
                 edge_embedding_nums=[],
                 edge_embedding_dims=[],
                 update_coors=True,
                 update_feats=True,
                 norm_feats=True,
                 norm_coors=True,
                 norm_coors_scale_init=1e-2,
                 dropout=0.,
                 coor_weights_clamp_value=None,
                 aggr="add",
                 global_linear_attn_every=0,
                 global_linear_attn_heads=8,
                 global_linear_attn_dim_head=64,
                 num_global_tokens=4,
                 recalc=0,):
        super().__init__()

        self.n_layers = n_layers

        # Embeddings? solve here
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        self.emb_layers = nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers = nn.ModuleList()

        # instantiate point and edge embedding layers

        for i in range(len(self.embedding_dims)):
            self.emb_layers.append(nn.Embedding(num_embeddings=embedding_nums[i],
                                                embedding_dim=embedding_dims[i]))
            feats_dim += embedding_dims[i] - 1

        for i in range(len(self.edge_embedding_dims)):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings=edge_embedding_nums[i],
                                                     embedding_dim=edge_embedding_dims[i]))
            edge_attr_dim += edge_embedding_dims[i] - 1
        # rest
        self.mpnn_layers = nn.ModuleList()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.edge_attr_dim = edge_attr_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.norm_coors_scale_init = norm_coors_scale_init
        self.update_feats = update_feats
        self.update_coors = update_coors
        self.dropout = dropout
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.recalc = recalc

        self.has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        self.global_linear_attn_every = global_linear_attn_every
        if self.has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        # instantiate layers
        for i in range(n_layers):
            layer = EGNN_Sparse(feats_dim=feats_dim,
                                pos_dim=pos_dim,
                                edge_attr_dim=edge_attr_dim,
                                m_dim=m_dim,
                                fourier_features=fourier_features,
                                soft_edge=soft_edge,
                                coors_tanh=coors_tanh,
                                norm_feats=norm_feats,
                                norm_coors=norm_coors,
                                norm_coors_scale_init=norm_coors_scale_init,
                                update_feats=update_feats,
                                update_coors=update_coors,
                                dropout=dropout,
                                coor_weights_clamp_value=coor_weights_clamp_value)

            # global attention case
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if is_global_layer:
                attn_layer = GlobalLinearAttention(dim=self.feats_dim,
                                                   heads=global_linear_attn_heads,
                                                   dim_head=global_linear_attn_dim_head)
                self.mpnn_layers.append(nn.ModuleList([layer, attn_layer]))
            # normal case
            else:
                self.mpnn_layers.append(layer)

    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Recalculate edge features every `self.recalc_edge` with the
            `recalc_edge` function if self.recalc_edge is set.

            * x: (N, pos_dim+feats_dim) will be unpacked into coors, feats.
        """
        # NODES - Embedd each dim to its target dimensions:
        x = embedd_token(x, self.embedding_dims, self.emb_layers)

        # regulates wether to embedd edges each layer
        edges_need_embedding = True
        for i, layer in enumerate(self.mpnn_layers):

            # EDGES - Embedd each dim to its target dimensions:
            if edges_need_embedding:
                edge_attr = embedd_token(edge_attr, self.edge_embedding_dims, self.edge_emb_layers)
                edges_need_embedding = False

            # attn tokens
            global_tokens = None
            if exists(self.global_tokens):
                unique, amounts = torch.unique(batch, return_counts)
                num_idxs = torch.cat([torch.arange(num_idxs_i) for num_idxs_i in amounts], dim=-1)
                global_tokens = self.global_tokens[num_idxs]

            # pass layers
            is_global_layer = self.has_global_attn and (i % self.global_linear_attn_every) == 0
            if not is_global_layer:
                x = layer(x, edge_index, edge_attr, batch=batch, size=bsize)
            else:
                # only pass feats to the attn layer
                x_attn = layer[0](x[:, self.pos_dim:], global_tokens)
                # merge attn-ed feats and coords
                x = torch.cat((x[:, :self.pos_dim], x_attn), dim=-1)
                x = layer[-1](x, edge_index, edge_attr, batch=batch, size=bsize)

            # recalculate edge info - not needed if last layer
            if self.recalc and ((i % self.recalc == 0) and not (i == len(self.mpnn_layers)-1)):
                edge_index, edge_attr, _ = recalc_edge(x)  #  returns attr, idx, any_other_info
                edges_need_embedding = True

        return x


class EGNNDynamics(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        module_cfg: DictConfig,
        diffusion_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        **kwargs
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

        # node and edge scalar embeddings
        self.node_embedding = nn.Linear(self.node_input_dims.scalar, self.node_dims.scalar)
        self.edge_embedding = nn.Linear(self.edge_input_dims.scalar, self.edge_dims.scalar)

        # EGNN layers
        self.egnn = EGNN_Sparse_Network(
            n_layers=model_cfg.num_encoder_layers,
            feats_dim=self.node_dims.scalar,
            edge_attr_dim=self.edge_dims.scalar
        )

        if diffusion_cfg.diffusion_target in NODE_FEATURE_DIFFUSION_TARGETS:
            # scalar node projection
            h_input_dim_without_self_conditioning = h_input_dim_ + h_input_conditioning_dim
            self.scalar_node_projection = nn.Linear(self.node_dims.scalar, h_input_dim_without_self_conditioning)

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
        batch.edge_index = GCPNetDynamics.get_fully_connected_edge_index(
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

        # begin EGNN forward pass #

        # centralize node positions to necessarily make them translation-invariant during the diffusion process
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True
        )

        # embed node and edge scalars
        batch.h, batch.e = self.node_embedding(batch.h) * batch.mask.float().unsqueeze(-1), self.edge_embedding(batch.e)

        # update node features and coordinates using a series of geometric message-passing layers
        egnn_out = self.egnn(
            x=torch.cat((batch.x, batch.h), dim=-1),
            edge_index=batch.edge_index,
            batch=batch.batch,
            edge_attr=batch.e,
            bsize=batch.num_graphs,
            recalc_edge=None,
            verbose=0
        )
        batch.x, h = egnn_out[:, :self.num_x_dims], egnn_out[:, self.num_x_dims:]
        batch.x, h = batch.x * batch.mask.float().unsqueeze(-1), h * batch.mask.float().unsqueeze(-1)

        # summarize scalar node features using a linear projection - note: the bias term might be non-zero
        h = self.scalar_node_projection(h) * batch.mask.float().unsqueeze(-1)

        # record final version of each feature in `Batch` object
        batch.h = h

        # end EGNN forward pass #

        vel = (batch.x - x_init) * batch.mask.float().unsqueeze(-1)  # use delta(x) to estimate noise
        h_final = batch.h

        # remove contextual features from output predictions if such features were originally provided
        if self.condition_on_context:
            h_final = h_final[:, :-self.num_context_node_features]
        if self.condition_on_time:
            h_final = h_final[:, :-1]  # note: here, the last dimension represents time

        # detect and nullify any invalid node position predictions
        if vel.isnan().any():
            log.warning(f"Detected NaN in `vel` -> resetting EGNN `vel` output for time step(s) {t} to zero.")
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
    _ = EGNNDynamics()
