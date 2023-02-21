"""
From https://github.com/ehoogeboom/e3_diffusion_for_molecules/
"""

import tempfile
import warnings
import torch
import pickle
import os
import openbabel
import numpy as np

from rdkit import Chem
from typing import Any, Dict, List, Optional, Tuple
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams

from src.datamodules.components.edm import dataset, get_bond_order_batch, get_bond_length_arrays
from src.datamodules.components.edm.datasets_config import get_dataset_info

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.models.components import write_xyz_file
from src.utils.pylogger import get_pylogger

patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


def compute_qm9_smiles(dataset_name, data_dir, remove_h):
    """

    :param dataset_name: QM9 or QM9_second_half
    :return:
    """
    log.info("\tConverting QM9 dataset to SMILES ...")

    class StaticArgs:
        def __init__(self, dataset, data_dir, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.data_dir = data_dir
            self.remove_h = remove_h
            self.include_charges = True
            self.subtract_thermo = True
            self.force_download = False
            self.create_pyg_graphs = False
            self.num_radials = 1
            self.device = "cpu"
            self.num_train = -1
            self.num_valid = -1
            self.num_test = -1
            self.shuffle = True
            self.drop_last = True
    args_dataset = StaticArgs(dataset_name, data_dir, remove_h)
    dataloaders, _ = dataset.retrieve_dataloaders(args_dataset)
    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []
    for i, data in enumerate(dataloaders["train"]):
        positions = data["positions"][0].view(-1, 3).numpy()
        one_hot = data["one_hot"][0].view(-1, n_types).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        charges = data["charges"][0].squeeze(-1)

        mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info, charges)
        mol = mol2smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        if i % 1000 == 0:
            log.info("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders["train"])))
    return mols_smiles


@typechecked
def retrieve_qm9_smiles(
    dataset_info: Dict[str, Any],
    data_dir: str
) -> List[str]:
    dataset_name = dataset_info["name"]
    if dataset_info["with_h"]:
        pickle_name = dataset_name
    else:
        pickle_name = dataset_name + "_noH"

    file_path = os.path.join(data_dir, dataset_name, f"{pickle_name}_smiles.pickle")
    try:
        with open(file_path, "rb") as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs(data_dir, exist_ok=True)
        except:
            pass
        qm9_smiles = compute_qm9_smiles(
            dataset_name,
            data_dir=data_dir,
            remove_h=not dataset_info["with_h"]
        )
        with open(file_path, "wb") as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles


#### New implementation ####

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]


class BasicMolecularMetrics(object):
    def __init__(
        self,
        dataset_info: Dict[str, Any],
        data_dir: str,
        dataset_smiles_list: Optional[np.ndarray] = None
    ):
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # retrieve dataset smiles only for the QM9 dataset currently
        if dataset_smiles_list is None and "QM9" in dataset_info["name"]:
            bonds = get_bond_length_arrays(self.dataset_info["atom_encoder"])
            self.dataset_info["bonds1"], self.dataset_info["bonds2"], self.dataset_info["bonds3"] = (
                bonds[0], bonds[1], bonds[2]
            )
            self.dataset_smiles_list = retrieve_qm9_smiles(self.dataset_info, data_dir)

    @typechecked
    def compute_validity(self, rdmols: List[Chem.RWMol]) -> Tuple[List[str], float]:
        valid = []
        for mol in rdmols:
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
        return valid, len(valid) / len(rdmols)

    @typechecked
    def compute_uniqueness(self, valid: List[str]) -> Tuple[List[str], float]:
        # note: `valid` is a list of SMILES strings
        return list(set(valid)), len(set(valid)) / len(valid)

    @typechecked
    def compute_novelty(self, unique: List[str]) -> Tuple[List[str], float]:
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    @typechecked
    def evaluate_rdmols(self, rdmols: List[Chem.RWMol], verbose: bool = True) -> List[float]:
        valid, validity = self.compute_validity(rdmols)
        if verbose:
            log.info(f"Validity over {len(rdmols)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            if verbose:
                log.info(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")
            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                if verbose:
                    log.info(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            uniqueness = 0.0
            novelty = 0.0
        return [validity, uniqueness, novelty]

    @typechecked
    def evaluate(
        self,
        generated: List[Tuple[torch.Tensor, ...]]
    ) -> List[float]:
        """
        note: `generated` is a list of pairs (`positions`: [n x 3], `atom_types`: n [int]; `charges`: n [int]);
        also note: the positions and atom types should already be masked.
        """
        rdmols = [build_molecule(*graph, self.dataset_info) for graph in generated]
        return self.evaluate_rdmols(rdmols)


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


@typechecked
def build_molecule(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any],
    charges: Optional[TensorType["num_nodes"]] = None,
    add_coords: bool = False,
    use_openbabel: bool = False
) -> Chem.RWMol:
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        charges: N
        dataset_info: dict
        add_coords: Add conformer to mol (always added if use_openbabel=True)
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types, dataset_info["atom_decoder"])
    else:
        mol = make_mol_edm(positions, atom_types, dataset_info, add_coords)

    return mol


@typechecked
def make_mol_openbabel(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    atom_decoder: Dict[int, str]
) -> Chem.RWMol:
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    atom_types = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        obConversion.WriteFile(ob_mol, tmp_file)

        # Read sdf file with RDKit
        mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    return mol


@typechecked
def make_mol_edm(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any],
    add_coords: bool
) -> Chem.RWMol:
    """
    Note: Equivalent to EDM's way of building RDKit molecules
    """
    n = len(positions)
    limit_bonds_to_one = "GEOM" in dataset_info["name"]

    # (X, A, E): atom_types, adjacency matrix, edge_types
    # X: N (int)
    # A: N x N (bool) -> (binary adjacency matrix)
    # E: N x N (int) -> (bond type, 0 if no bond)
    pos = positions.unsqueeze(0)  # add batch dim
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1)  # remove batch dim & flatten
    atoms1, atoms2 = torch.cartesian_prod(atom_types, atom_types).T
    E_full = get_bond_order_batch(
        atoms1, atoms2, dists, dataset_info, limit_bonds_to_one=limit_bonds_to_one
    ).view(n, n)
    E = torch.tril(E_full, diagonal=-1)  # Warning: the graph should be DIRECTED
    A = E.bool()
    X = atom_types

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(),
                    bond_dict[E[bond[0], bond[1]].item()])

    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (positions[i, 0].item(),
                                     positions[i, 1].item(),
                                     positions[i, 2].item()))
        mol.AddConformer(conf)

    return mol


@typechecked
def process_molecule(
    rdmol: Chem.Mol,
    add_hydrogens: bool = False,
    sanitize: bool = False,
    relax_iter: int = 0,
    largest_frag: bool = False
) -> Optional[Chem.Mol]:
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: RDKit molecule
        add_hydrogens: whether to add hydrogen atoms to the generated molecule
        sanitize: whether to sanitize molecules
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: whether to filter out the largest fragment in a set of disjoint molecules
    Returns:
        RDKit molecule or `None` if it does not pass the filters
    """
    # create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. '
                          'Returning None.')
            return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol


@typechecked
def uff_relax(mol: Chem.Mol, max_iter: int = 200) -> bool:
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    convergence_status = UFFOptimizeMolecule(mol, maxIters=max_iter)
    more_iterations_required = convergence_status == 1
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


if __name__ == "__main__":
    smiles_mol = "C1CCC1"
    log.info("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    log.info("Block mol:")
    log.info(block_mol)
