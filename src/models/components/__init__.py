# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import os
import imageio
import random
import torch
import matplotlib
import wandb
import numpy as np
import matplotlib.pyplot as plt
import prody as pr

from torch import nn
from torch_scatter import scatter
from torch_geometric.data import Batch
from matplotlib.axes._subplots import Axes
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from wandb.sdk.wandb_run import Run
from pathlib import Path
from rdkit import Chem

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.datamodules.components.edm import get_bond_order
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    from pymol import cmd
except ImportError:
    log.warning("PyMOL not found.")

patch_typeguard()  # use before @typechecked

pr.confProDy(verbosity="none")

matplotlib.use("Agg")


@typechecked
def centralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    node_mask: Optional[TensorType["batch_num_nodes"]] = None,
    edm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        # derive centroid of each batch element
        if edm:
            masked_max_abs_value = batch[key][~node_mask].abs().sum().item()
            assert masked_max_abs_value < 1e-5, f"Masked CoG error {masked_max_abs_value} is too high"

            num_entities = scatter(
                node_mask.float(),
                batch_index,
                dim=0,
                reduce="sum"
            ).unsqueeze(-1)
            entities_sum = scatter(
                batch[key],
                batch_index,
                dim=0,
                reduce="sum"
            )
            entities_centroid = entities_sum / num_entities
        else:
            entities_centroid = scatter(
                batch[key][node_mask],
                batch_index[node_mask],
                dim=0,
                reduce="mean"
            )  # e.g., [batch_size, 3]

        # center entities using corresponding centroids
        if edm:
            entities_centered = batch[key] - (entities_centroid[batch_index] * node_mask.float().unsqueeze(-1))
        else:
            masked_values = (
                torch.zeros_like(batch[key])
                if edm
                else torch.ones_like(batch[key]) * torch.inf
            )
            values = batch[key][node_mask]
            masked_values[node_mask] = (values - entities_centroid[batch_index][node_mask])
            entities_centered = masked_values

    else:
        # derive centroid of each batch element, and center entities using corresponding centroids
        entities_centroid = scatter(batch[key], batch_index, dim=0, reduce="mean")  # e.g., [batch_size, 3]
        entities_centered = batch[key] - entities_centroid[batch_index]

    return entities_centroid, entities_centered


@typechecked
def decentralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    entities_centroid: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    edm: bool = False
) -> torch.Tensor:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        if edm:
            entities_centered = batch[key] + (entities_centroid[batch_index] * node_mask.float().unsqueeze(-1))
        else:
            masked_values = torch.ones_like(batch[key]) * torch.inf
            masked_values[node_mask] = (batch[key][node_mask] + entities_centroid[batch_index])
            entities_centered = masked_values
    else:
        entities_centered = batch[key] + entities_centroid[batch_index]
    return entities_centered


@typechecked
def localize(
    x: TensorType["batch_num_nodes", 3],
    edge_index: TensorType[2, "batch_num_edges"],
    norm_x_diff: bool = True,
    node_mask: Optional[torch.Tensor] = None
) -> TensorType["batch_num_edges", 3, 3]:
    row, col = edge_index[0], edge_index[1]

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

        x_diff = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        x_diff[edge_mask] = x[row][edge_mask] - x[col][edge_mask]

        x_cross = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        x_cross[edge_mask] = torch.cross(x[row][edge_mask], x[col][edge_mask])
    else:
        x_diff = x[row] - x[col]
        x_cross = torch.cross(x[row], x[col])

    if norm_x_diff:
        # derive and apply normalization factor for `x_diff`
        if node_mask is not None:
            norm = torch.ones((edge_index.shape[1], 1), device=x_diff.device)
            norm[edge_mask] = (
                torch.sqrt(torch.sum((x_diff[edge_mask] ** 2), dim=1).unsqueeze(1))
            ) + 1
        else:
            norm = torch.sqrt(torch.sum((x_diff) ** 2, dim=1).unsqueeze(1)) + 1
        x_diff = x_diff / norm

        # derive and apply normalization factor for `x_cross`
        if node_mask is not None:
            cross_norm = torch.ones((edge_index.shape[1], 1), device=x_cross.device)
            cross_norm[edge_mask] = (
                torch.sqrt(torch.sum((x_cross[edge_mask]) ** 2, dim=1).unsqueeze(1))
            ) + 1
        else:
            cross_norm = (torch.sqrt(torch.sum((x_cross) ** 2, dim=1).unsqueeze(1))) + 1
        x_cross = x_cross / cross_norm

    if node_mask is not None:
        x_vertical = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        x_vertical[edge_mask] = torch.cross(x_diff[edge_mask], x_cross[edge_mask])
    else:
        x_vertical = torch.cross(x_diff, x_cross)

    f_ij = torch.cat((x_diff.unsqueeze(1), x_cross.unsqueeze(1), x_vertical.unsqueeze(1)), dim=1)
    return f_ij


@typechecked
def scalarize(
    vector_rep: TensorType["batch_num_entities", 3, 3],
    edge_index: TensorType[2, "batch_num_edges"],
    frames: TensorType["batch_num_edges", 3, 3],
    node_inputs: bool,
    dim_size: int,
    node_mask: Optional[TensorType["batch_num_nodes"]] = None
) -> TensorType["effective_batch_num_entities", 9]:
    row, col = edge_index[0], edge_index[1]

    # gather source node features for each `entity` (i.e., node or edge)
    # note: edge inputs are already ordered according to source nodes
    vector_rep_i = vector_rep[row] if node_inputs else vector_rep

    # project equivariant values onto corresponding local frames
    if vector_rep_i.ndim == 2:
        vector_rep_i = vector_rep_i.unsqueeze(-1)
    elif vector_rep_i.ndim == 3:
        vector_rep_i = vector_rep_i.transpose(-1, -2)

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]
        local_scalar_rep_i = torch.zeros((edge_index.shape[1], 3, 3), device=edge_index.device)
        local_scalar_rep_i[edge_mask] = torch.matmul(
            frames[edge_mask], vector_rep_i[edge_mask]
        )
        local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
    else:
        local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(-1, -2)

    # reshape frame-derived geometric scalars
    local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

    if node_inputs:
        # for node inputs, summarize all edge-wise geometric scalars using an average
        return scatter(
            local_scalar_rep_i,
            # summarize according to source node indices due to the directional nature of GCP2's equivariant frames
            row,
            dim=0,
            dim_size=dim_size,
            reduce="mean"
        )

    return local_scalar_rep_i


@typechecked
def vectorize(
    gate: TensorType["batch_num_entities", 9],
    edge_index: TensorType[2, "batch_num_edges"],
    frames: TensorType["batch_num_edges", 3, 3],
    node_inputs: bool,
    dim_size: int,
    node_mask: Optional[TensorType["batch_num_nodes"]] = None
) -> TensorType["effective_batch_num_entities", 3, 3]:
    row, col = edge_index

    frames = frames.reshape(frames.shape[0], 1, 9)
    x_diff, x_cross, x_vertical = frames[:, :, :3].squeeze(
    ), frames[:, :, 3:6].squeeze(), frames[:, :, 6:].squeeze()

    # gather source node features for each `entity` (i.e., node or edge)
    gate = gate[row] if node_inputs else gate  # note: edge inputs are already ordered according to source nodes

    # derive edge mask if provided node mask
    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

    # use invariant scalar features to derive new vector features using each neighboring node
    gate_vector = torch.zeros_like(gate)
    for i in range(0, gate.shape[-1], 3):
        if node_mask is not None:
            gate_vector[edge_mask, i:i + 3] = (
                gate[edge_mask, i:i + 1] * x_diff[edge_mask]
                + gate[edge_mask, i + 1:i + 2] * x_cross[edge_mask]
                + gate[edge_mask, i + 2:i + 3] * x_vertical[edge_mask]
            )
        else:
            gate_vector[:, i:i + 3] = (
                gate[:, i:i + 1] * x_diff
                + gate[:, i + 1:i + 2] * x_cross
                + gate[:, i + 2:i + 3] * x_vertical
            )
    gate_vector = gate_vector.reshape(gate_vector.shape[0], 3, 3)

    # for node inputs, summarize all edge-wise geometric vectors using an average
    if node_inputs:
        return scatter(
            gate_vector,
            # summarize according to source node indices due to the directional nature of GCP2's equivariant frames
            row,
            dim=0,
            dim_size=dim_size,
            reduce="mean"
        )

    return gate_vector


@typechecked
def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True
):
    norm = torch.sum(x ** 2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps


@typechecked
def is_identity(nonlinearity: Optional[Union[Callable, nn.Module]] = None):
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)


@typechecked
def norm_no_nan(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True
):
    """
    From https://github.com/drorlab/gvp-pytorch

    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


@typechecked
def num_nodes_to_batch_index(
    num_samples: int,
    num_nodes: Union[int, TensorType["batch_size"]],
    device: Union[torch.device, str]
) -> TensorType["batch_num_nodes"]:
    assert isinstance(num_nodes, int) or len(num_nodes) == num_samples
    sample_inds = torch.arange(num_samples, device=device)
    return torch.repeat_interleave(sample_inds, num_nodes)


@typechecked
def save_xyz_file(
    path: str,
    positions: TensorType["batch_num_nodes", 3],
    one_hot: TensorType["batch_num_nodes", "num_atom_types"],
    charges: torch.Tensor,  # TODO: incorporate charges within saved XYZ file
    dataset_info: Dict[str, Any],
    id_from: int = 0,
    name: str = "molecule",
    batch_index: Optional[TensorType["batch_num_nodes"]] = None
):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if batch_index is None:
        batch_index = torch.zeros(len(one_hot))

    for batch_i in torch.unique(batch_index):
        current_batch_index = (batch_index == batch_i)
        num_atoms = int(torch.sum(current_batch_index).item())
        f = open(path + name + "_" + "%03d.xyz" % (batch_i + id_from), "w")
        f.write("%d\n\n" % num_atoms)
        atoms = torch.argmax(one_hot[current_batch_index], dim=-1)
        batch_pos = positions[current_batch_index]
        for atom_i in range(num_atoms):
            atom = atoms[atom_i]
            atom = dataset_info["atom_decoder"][atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, batch_pos[atom_i, 0], batch_pos[atom_i, 1], batch_pos[atom_i, 2]))
        f.close()


@typechecked
def write_xyz_file(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    filename: str
):
    out = f"{len(positions)}\n\n"
    assert len(positions) == len(atom_types)
    for i in range(len(positions)):
        out += f"{atom_types[i]} {positions[i, 0]:.3f} {positions[i, 1]:.3f} {positions[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


@typechecked
def write_sdf_file(sdf_path: Path, molecules: List[Chem.Mol], verbose: bool = True):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)
    if verbose:
        log.info(f"Wrote generated molecules to SDF file {sdf_path}")


@typechecked
def load_molecule_xyz(
    file: str,
    dataset_info: Dict[str, Any]
) -> Tuple[
    TensorType["num_nodes", 3],
    TensorType["num_nodes", "num_atom_types"]
]:
    with open(file, encoding="utf8") as f:
        num_atoms = int(f.readline())
        one_hot = torch.zeros(num_atoms, len(dataset_info["atom_decoder"]))
        positions = torch.zeros(num_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(num_atoms):
            atom = atoms[i].split(" ")
            atom_type = atom[0]
            one_hot[i, dataset_info["atom_encoder"][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot


@typechecked
def load_files_with_ext(path: str, ext: str, shuffle: bool = True) -> List[str]:
    files = glob.glob(path + f"/*.{ext}")
    if shuffle:
        random.shuffle(files)
    return files


@typechecked
def visualize_mol(
    path: str,
    dataset_info: Dict[str, Any],
    max_num: int = 25,
    wandb_run: Optional[Run] = None,
    spheres_3d: bool = False,
    mode: str = "molecule",
    verbose: bool = True
):
    files = load_files_with_ext(path, ext="xyz")[0: max_num]
    for file in files:
        positions, one_hot = load_molecule_xyz(file, dataset_info)
        atom_types = torch.argmax(one_hot, dim=-1).numpy()
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
        dists = dists[dists > 0]

        if verbose:
            log.info(f"Average distance between atoms: {dists.mean().item()}")

        plot_data3d(
            positions=positions,
            atom_types=atom_types,
            dataset_info=dataset_info,
            save_path=f"{file[:-4]}.png",
            spheres_3d=spheres_3d
        )

        if wandb_run is not None:
            # log image(s) via WandB
            path = f"{file[:-4]}.png"
            im = plt.imread(path)
            wandb_run.log({mode: [wandb.Image(im, caption=path)]})


@typechecked
def visualize_mol_chain(
    path: str,
    dataset_info: Dict[str, Any],
    wandb_run: Optional[Run] = None,
    spheres_3d: bool = False,
    mode: str = "chain",
    verbose: bool = True
):
    files = load_files_with_ext(path, ext="xyz")
    files = sorted(files)
    save_paths = []

    for file in files:
        positions, one_hot = load_molecule_xyz(file, dataset_info=dataset_info)

        atom_types = torch.argmax(one_hot, dim=-1).numpy()
        fn = f"{file[:-4]}.png"
        plot_data3d(
            positions=positions,
            atom_types=atom_types,
            dataset_info=dataset_info,
            save_path=fn,
            spheres_3d=spheres_3d,
            alpha=1.0
        )

        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = os.path.join(dirname, "output.gif")

    if verbose:
        log.info(f"Creating GIF with {len(imgs)} images")

    # add the last frame 10 times so that the final result remains temporally
    # imgs.extend([imgs[-1]] * 10)

    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb_run is not None:
        wandb_run.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


@typechecked
def draw_sphere(
    ax: plt.axis,
    x: float,
    y: float,
    z: float,
    size: float,
    color: str,
    alpha: float
):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x + xs,
        y + ys,
        z + zs,
        rstride=2,
        cstride=2,
        color=color,
        linewidth=0,
        alpha=alpha
    )


@typechecked
def plot_molecule(
    ax: Axes,
    positions: TensorType["num_nodes", 3],
    atom_types: np.ndarray,
    alpha: float,
    spheres_3d: bool,
    hex_bg_color: str,
    dataset_info: Dict[str, Any]
):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    colors_dic = np.array(dataset_info["colors_dic"])
    radius_dic = np.array(dataset_info["radius_dic"])
    area_dic = 1500 * radius_dic ** 2

    areas = area_dic[atom_types]
    radii = radius_dic[atom_types]
    colors = colors_dic[atom_types]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha,
                   c=colors)  # , linewidths=2, edgecolors="#FFFFFF")

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))

            s = (atom_types[i], atom_types[j])

            draw_edge_int = get_bond_order(
                dataset_info["atom_decoder"][s[0]],
                dataset_info["atom_decoder"][s[1]],
                dist
            )
            line_width = 2

            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    # linewidth_factor = draw_edge_int  # note: proportional to number of edges
                    linewidth_factor = 1
                ax.plot(
                    [x[i], x[j]],
                    [y[i], y[j]],
                    [z[i], z[j]],
                    linewidth=line_width * linewidth_factor,
                    c=hex_bg_color,
                    alpha=alpha
                )


@typechecked
def plot_data3d(
    positions: TensorType["num_nodes", 3],
    atom_types: np.ndarray,
    dataset_info: Dict[str, Any],
    camera_elev: int = 0,
    camera_azim: int = 0,
    save_path: str = None,
    spheres_3d: bool = False,
    bg: str = "black",
    alpha: float = 1.0
):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = "#FFFFFF" if bg == "black" else "#666666"

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("auto")
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == "black":
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)

    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == "black":
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    plot_molecule(
        ax=ax,
        positions=positions,
        atom_types=atom_types,
        alpha=alpha,
        spheres_3d=spheres_3d,
        hex_bg_color=hex_bg_color,
        dataset_info=dataset_info
    )

    if "GEOM" in dataset_info["name"]:
        max_value = positions.abs().max().item()

        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        max_value = positions.abs().max().item()

        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype("uint8")
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


class ScalarVector(tuple):
    """
    From https://github.com/sarpaykent/GBPNet
    """
    def __new__(cls, scalar, vector):
        return tuple.__new__(cls, (scalar, vector))

    def __getnewargs__(self):
        return self.scalar, self.vector

    @property
    def scalar(self):
        return self[0]

    @property
    def vector(self):
        return self[1]

    # Element-wise addition
    def __add__(self, other):
        if isinstance(other, tuple):
            scalar_other = other[0]
            vector_other = other[1]
        else:
            scalar_other = other.scalar
            vector_other = other.vector

        return ScalarVector(self.scalar + scalar_other, self.vector + vector_other)

    # Element-wise multiplication or scalar multiplication
    def __mul__(self, other):
        if isinstance(other, tuple):
            other = ScalarVector(other[0], other[1])

        if isinstance(other, ScalarVector):
            return ScalarVector(self.scalar * other.scalar, self.vector * other.vector)
        else:
            return ScalarVector(self.scalar * other, self.vector * other)

    def concat(self, others, dim=-1):
        dim %= len(self.scalar.shape)
        s_args, v_args = list(zip(*(self, *others)))
        return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

    def flatten(self):
        flat_vector = torch.reshape(self.vector, self.vector.shape[:-2] + (3 * self.vector.shape[-2],))
        return torch.cat((self.scalar, flat_vector), dim=-1)

    @staticmethod
    def recover(x, vector_dim):
        v = torch.reshape(x[..., -3 * vector_dim:], x.shape[:-1] + (vector_dim, 3))
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def vs(self):
        return self.scalar, self.vector

    def idx(self, idx):
        return ScalarVector(self.scalar[idx], self.vector[idx])

    def repeat(self, n, c=1, y=1):
        return ScalarVector(self.scalar.repeat(n, c), self.vector.repeat(n, y, c))

    def clone(self):
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    def mask(self, node_mask: TensorType["num_nodes"]):
        return ScalarVector(
            self.scalar * node_mask[:, None],
            self.vector * node_mask[:, None, None]
        )

    def __setitem__(self, key, value):
        self.scalar[key] = value.scalar
        self.vector[key] = value.vector

    def __repr__(self):
        return f"ScalarVector({self.scalar}, {self.vector})"


class VectorDropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GCPDropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float, use_gcp_dropout: bool = True):
        super().__init__()
        self.scalar_dropout = nn.Dropout(drop_rate) if use_gcp_dropout else nn.Identity()
        self.vector_dropout = VectorDropout(drop_rate) if use_gcp_dropout else nn.Identity()

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))


class GCPLayerNorm(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, dims: ScalarVector, eps: float = 1e-8, use_gcp_norm: bool = True):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dims) if use_gcp_norm else nn.Identity()
        self.use_gcp_norm = use_gcp_norm
        self.eps = eps

    @staticmethod
    def norm_vector(v: torch.Tensor, use_gcp_norm: bool = True, eps: float = 1e-8) -> torch.Tensor:
        v_norm = v
        if use_gcp_norm:
            vector_norm = torch.clamp(torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps)
            vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
            v_norm = v / vector_norm
        return v_norm

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        return ScalarVector(self.scalar_norm(s), self.norm_vector(v, use_gcp_norm=self.use_gcp_norm, eps=self.eps))
