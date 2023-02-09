"""
From https://raw.githubusercontent.com/arneschneuing/DiffSBDD/
"""

import os
import torch
import subprocess

import numpy as np

import src.datamodules.components.edm.constants as edm_constants

from typing import Any, Dict, List, Literal, Tuple, Union

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.utils.pylogger import get_pylogger

patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


@typechecked
def get_bond_length_arrays(atom_mapping: Dict[str, int]) -> List[np.ndarray]:
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(edm_constants, f'bonds{i + 1}')
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


@typechecked
def get_bond_order(
    atom1: str,
    atom2: str,
    distance: Union[float, np.float32]
) -> int:
    distance = 100 * distance  # We change the metric
    if atom1 in edm_constants.bonds3 and atom2 in edm_constants.bonds3[atom1] and distance < edm_constants.bonds3[atom1][atom2] + edm_constants.margin3:
        return 3  # triple bond
    if atom1 in edm_constants.bonds2 and atom2 in edm_constants.bonds2[atom1] and distance < edm_constants.bonds2[atom1][atom2] + edm_constants.margin2:
        return 2  # double bond
    if atom1 in edm_constants.bonds1 and atom2 in edm_constants.bonds1[atom1] and distance < edm_constants.bonds1[atom1][atom2] + edm_constants.margin1:
        return 1  # single bond
    return 0      # no bond


@typechecked
def get_bond_order_batch(
    atoms1: TensorType["num_pairwise_atoms"],
    atoms2: TensorType["num_pairwise_atoms"],
    distances: TensorType["num_pairwise_atoms"],
    dataset_info: Dict[str, Any],
    limit_bonds_to_one: bool = False
) -> TensorType["num_pairwise_atoms"]:
    distances = 100 * distances  # note: we change the metric

    bonds1 = torch.tensor(dataset_info['bonds1'])
    bonds2 = torch.tensor(dataset_info['bonds2'])
    bonds3 = torch.tensor(dataset_info['bonds3'])

    bond_types = torch.zeros_like(atoms1)  # note: `0` indicates no bond

    # single bond
    bond_types[distances < (bonds1[atoms1, atoms2] + edm_constants.margin1)] = 1
    # double bond (note: already assigned single bonds will be overwritten)
    bond_types[distances < (bonds2[atoms1, atoms2] + edm_constants.margin2)] = 2
    # triple bond
    bond_types[distances < (bonds3[atoms1, atoms2] + edm_constants.margin3)] = 3

    if limit_bonds_to_one:
        # e.g., for datasets such as GEOM-Drugs
        bond_types[bond_types > 1] = 1

    return bond_types


@typechecked
def check_molecular_stability(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any],
    verbose: bool = False
) -> Tuple[bool, int, int]:
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    atom_decoder = dataset_info['atom_decoder']
    n = len(positions)

    dists = torch.cdist(positions, positions, p=2.0).reshape(-1)
    atoms1, atoms2 = torch.meshgrid(atom_types, atom_types, indexing="xy")
    atoms1, atoms2 = atoms1.reshape(-1), atoms2.reshape(-1)
    order = get_bond_order_batch(atoms1, atoms2, dists, dataset_info).numpy().reshape(n, n)
    np.fill_diagonal(order, 0)  # mask out diagonal (i.e., self) bonds
    nr_bonds = np.sum(order, axis=1)

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_types, nr_bonds):
        possible_bonds = edm_constants.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and verbose:
            log.info("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == n
    return molecule_stable, nr_stable_bonds, n
