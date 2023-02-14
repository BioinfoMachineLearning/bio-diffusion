# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import logging
import torch_geometric

from typing import Any, Dict, Optional, Tuple, Union
from torch_geometric.data import Data, Dataset, Batch

from src.datamodules.components.helper import _normalize
from src.datamodules.components.protein_graph_dataset import ProteinGraphDataset

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@typechecked
def _edge_features(
    batch: Batch,
    coords_key: str = "x"
) -> Tuple[
    TensorType["num_edges", "num_edge_scalar_features"],
    TensorType["num_edges", "num_edge_vector_features", 3]
]:
    coords = batch[coords_key]
    E_vectors = coords[batch.edge_index[0]] - coords[batch.edge_index[1]]
    radial = torch.sum(E_vectors ** 2, dim=1).unsqueeze(-1)

    edge_s = radial
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


@typechecked
def _node_features(
    batch: Batch,
    coords_key: str = "x",
    edm_sampling: bool = False
) -> Tuple[
    Union[
        Dict[
            str,
            Union[
                TensorType["num_nodes", "num_atom_types"],
                torch.Tensor  # note: for when `include_charges=False`
            ]
        ],
        TensorType["num_nodes", "num_node_scalar_features"],
        Optional[torch.Tensor]
    ],
    TensorType["num_nodes", "num_node_vector_features", 3]
]:
    # construct invariant node features
    if hasattr(batch, "h"):
        node_s = batch.h
    elif edm_sampling:
        node_s = None
    else:
        node_s = {"categorical": batch.one_hot, "integer": batch.charges}
        node_s["categorical"], node_s["integer"] = (
            map(torch.nan_to_num, (node_s["categorical"], node_s["integer"]))
        )

    # build equivariant node features
    coords = batch[coords_key]
    orientations = ProteinGraphDataset._orientations(coords)
    node_v = torch.nan_to_num(orientations)

    return node_s, node_v


class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Geometric Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    create_pyg_graphs: bool, optional
        If true, return PyTorch Geometric graphs when requesting dataset examples
    num_radials: int, optional
        Number of radial (distance) features to compute for each edge
    device: Union[torch.device, str], optional
        On which device to create graph features
    remove_zero_charge_molecules: bool, optional
        Whether to filter out from the dataset molecules with a total of zero charge
    """

    def __init__(
        self,
        data,
        included_species=None,
        num_pts=-1,
        shuffle=True,
        subtract_thermo=True,
        create_pyg_graphs=True,
        num_radials=1,
        device="cpu",
        remove_zero_charge_molecules=True
    ):

        self.data = data
        self.num_radials = num_radials
        self.device = device

        if remove_zero_charge_molecules:
            nonzero_charge_molecule_mask = self.data["charges"].sum(-1) > 0
            self.data = {key: val[nonzero_charge_molecule_mask] for key, val in self.data.items()}

        if num_pts < 0:
            self.num_pts = len(data["charges"])
        else:
            if num_pts > len(data["charges"]):
                logging.warning("Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!".format(
                    num_pts, len(data["charges"])))
                self.num_pts = len(data["charges"])
            else:
                self.num_pts = num_pts

        if included_species is None:
            # i.e., if included species is not specified
            included_species = torch.unique(self.data["charges"], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split("_")[0] for key in self.data.keys() if key.endswith("_thermo")]
            if len(thermo_targets) == 0:
                logging.warning("No thermochemical targets included! Try reprocessing dataset with --force-download!")
            else:
                logging.info("Removing thermochemical energy from targets {}".format(" ".join(thermo_targets)))
            for key in thermo_targets:
                self.data[key] -= self.data[key + "_thermo"].to(self.data[key].dtype)

        self.included_species = included_species

        self.data["one_hot"] = self.data["charges"].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {"num_species": self.num_species, "max_charge": self.max_charge}

        # get a dictionary of statistics for all properties that are one-dimensional tensors
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data["charges"]))[:self.num_pts]
        else:
            self.perm = None

        # determine which featurization method to use for requested dataset examples
        self.featurize_as_graph = (
            self._featurize_as_graph if create_pyg_graphs else lambda x: x
        )

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val)
                      is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    @typechecked
    def _featurize_as_graph(self, molecule: Dict[str, Any], dtype: torch.dtype = torch.float32) -> Data:
        with torch.no_grad():
            index = molecule["index"].unsqueeze(-1)
            coords = molecule["positions"].type(dtype)

            mask = molecule["charges"] > 0
            coords[~mask] = 0.0  # ensure missing nodes are assigned no edges

            # derive edges without self-loops
            edge_mask = mask.unsqueeze(0) * mask.unsqueeze(1)
            diag_mask = ~torch.eye(edge_mask.shape[0], dtype=torch.bool, device=edge_mask.device)
            edge_mask *= diag_mask
            edge_index = torch.stack(torch.where(edge_mask))

            one_hot = molecule["one_hot"].type(torch.float32)
            charges = molecule["charges"].type(torch.float32)

            conditional_properties = {
                key: value.reshape(1).type(dtype) for key, value in molecule.items()
                if key not in ["num_atoms", "charges", "positions", "index", "one_hot", "atom_mask"]
            }
            return torch_geometric.data.Data(
                one_hot=one_hot,
                charges=charges,
                x=coords,
                index=index,
                edge_index=edge_index,
                mask=mask,
                **conditional_properties
            )

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return self.featurize_as_graph(
            {key: val[idx] for key, val in self.data.items()}
        )
