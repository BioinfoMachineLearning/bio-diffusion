"""
From https://github.com/ehoogeboom/e3_diffusion_for_molecules/
"""

import msgpack
import os
import torch
import argparse
import numpy as np

from typing import Any, Dict
import torch_geometric
from torch_geometric.data import Data, Dataset

from torch.utils.data import BatchSampler, DataLoader as TorchDataLoader, SequentialSampler
from torch_geometric.loader.dataloader import DataLoader as PyGDataLoader

from src.datamodules.components.edm import collate as qm9_collate

from torchtyping import patch_typeguard
from typeguard import typechecked

from src.utils.pylogger import get_pylogger

patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


def extract_conformers(args):
    drugs_file = os.path.join(args.data_dir, args.data_file)
    save_file = f"GEOM_drugs_{'no_h_' if args.remove_h else ''}{args.conformations}"
    smiles_list_file = "GEOM_drugs_smiles.txt"
    number_atoms_file = f"GEOM_drugs_n_{'no_h_' if args.remove_h else ''}{args.conformations}"

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    for i, drugs_1k in enumerate(unpacker):
        log.info(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            all_smiles.append(smiles)
            conformers = all_info["conformers"]
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer["totalenergy"])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:args.conformations]
            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer["xyz"]).astype(float)        # n x 4
                if args.remove_h:
                    mask = coords[:, 0] != 1.0
                    coords = coords[mask]
                n = coords.shape[0]
                all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                dataset_conformers.append(id_coords)
                mol_id += 1

    log.info("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    log.info("Total number of atoms in the dataset", dataset.shape[0])
    log.info("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    np.save(os.path.join(args.data_dir, save_file), dataset)
    # Save SMILES
    with open(os.path.join(args.data_dir, smiles_list_file), "w") as f:
        for s in all_smiles:
            f.write(s)
            f.write("\n")

    # Save number of atoms per conformation
    np.save(os.path.join(args.data_dir, number_atoms_file), all_number_atoms)
    log.info("Dataset processed.")


def load_split_data(conformation_file, val_proportion=0.1, test_proportion=0.1,
                    filter_size=None):
    from pathlib import Path
    path = Path(conformation_file)
    base_path = path.parent.absolute()

    # base_path = os.path.dirname(conformation_file)
    all_data = np.load(conformation_file)  # 2d array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)

    # Filter based on molecule size.
    if filter_size is not None:
        # Keep only molecules <= filter_size
        data_list = [molecule for molecule in data_list
                     if molecule.shape[0] <= filter_size]

        assert len(data_list) > 0, "No molecules left after filter."

    # CAREFUL! Only for first time run:
    # perm = np.random.permutation(len(data_list)).astype("int32")
    # log.warning("Currently taking a random permutation for "
    #       "train/val/test partitions, this needs to be fixed for"
    #       "reproducibility.")
    # assert not os.path.exists(os.path.join(base_path, "GEOM_permutation.npy"))
    # np.save(os.path.join(base_path, "GEOM_permutation.npy"), perm)
    # del perm

    perm = np.load(os.path.join(base_path, "GEOM_permutation.npy"))
    data_list = np.array([data_list[i] for i in perm], dtype=object)

    num_mol = len(data_list)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    val_data, test_data, train_data = np.split(data_list, [val_index, test_index])
    return train_data, val_data, test_data


class GeomDrugsDataset(Dataset):
    def __init__(
        self,
        data_list,
        transform=None,
        create_pyg_graphs=True,
        num_radials=1,
        device="cpu"
    ):
        """
        Args:
            data_list (np.ndarray): Nested NumPy array of features for input samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            create_pyg_graphs (bool, optional): If true, return PyTorch Geometric graphs
                when requesting dataset examples.
            num_radials: (int, optional)
                Number of radial (distance) features to compute for each edge.
            device: (Union[torch.device, str], optional)
                On which device to create graph features.
        """
        self.transform = transform
        self.num_radials = num_radials
        self.device = device

        # Sort the data list by size
        lengths = [s.shape[0] for s in data_list]
        argsort = np.argsort(lengths)               # Sort by decreasing size
        self.data_list = [data_list[i] for i in argsort]
        # Store indices where the size changes
        self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]

        # determine which featurization method to use for requested dataset examples
        self.featurize_as_graph = (
            self._featurize_as_graph if create_pyg_graphs else lambda x: x
        )

    @typechecked
    def _featurize_as_graph(self, molecule: Dict[str, Any], dtype: torch.dtype = torch.float32) -> Data:
        with torch.no_grad():
            index = molecule["index"].unsqueeze(-1)
            coords = molecule["positions"].type(dtype)

            mask = molecule["atom_mask"].to(torch.bool)
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
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)

        sample["index"] = torch.tensor(idx, dtype=torch.long)
        return self.featurize_as_graph(sample)


class CustomBatchSampler(BatchSampler):
    """ Creates batches where all sets have the same size. """

    def __init__(self, sampler, batch_size, drop_last, split_indices):
        super().__init__(sampler, batch_size, drop_last)
        self.split_indices = split_indices

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size or idx + 1 in self.split_indices:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        count = 0
        batch = 0
        for idx in self.sampler:
            batch += 1
            if batch == self.batch_size or idx + 1 in self.split_indices:
                count += 1
                batch = 0
        if batch > 0 and not self.drop_last:
            count += 1
        return count


def collate_fn(batch):
    batch = {prop: qm9_collate.batch_stack([mol[prop] for mol in batch])
             for prop in batch[0].keys()}

    atom_mask = batch["atom_mask"]

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool,
                           device=edge_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch["edge_mask"] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    return batch


class GeomDrugsTorchDataLoader(TorchDataLoader):
    def __init__(self, sequential, dataset, batch_size, shuffle, drop_last=False, **kwargs):
        self.sequential = sequential

        if self.sequential:
            # This goes over the data sequentially, advantage is that it takes
            # less memory for smaller molecules, but disadvantage is that the
            # model sees very specific orders of data.
            assert not shuffle
            sampler = SequentialSampler(dataset)
            batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last,
                                               dataset.split_indices)
            super().__init__(dataset, batch_sampler=batch_sampler)
        else:
            # Dataloader goes through data randomly and pads the molecules to
            # the largest molecule size.
            super().__init__(
                dataset, batch_size, shuffle=shuffle,
                collate_fn=collate_fn, drop_last=drop_last
            )


class GeomDrugsPyGDataLoader(PyGDataLoader):
    def __init__(self, sequential, dataset, batch_size, shuffle, drop_last=False, **kwargs):
        self.sequential = sequential

        if self.sequential:
            # This goes over the data sequentially, advantage is that it takes
            # less memory for smaller molecules, but disadvantage is that the
            # model sees very specific orders of data.
            assert not shuffle
            sampler = SequentialSampler(dataset)
            batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last,
                                               dataset.split_indices)
            super().__init__(dataset, batch_sampler=batch_sampler)
        else:
            # Dataloader goes through data randomly and pads the molecules to
            # the largest molecule size.
            super().__init__(
                dataset, batch_size,
                shuffle=shuffle, drop_last=drop_last
            )


class GeomDrugsTransform(object):
    def __init__(self, dataset_info, include_charges, device, sequential):
        self.atomic_number_list = torch.Tensor(dataset_info["atomic_nb"])[None, :]
        self.device = device
        self.include_charges = include_charges
        self.sequential = sequential

    def __call__(self, data):
        n = data.shape[0]
        new_data = {}
        new_data["positions"] = torch.from_numpy(data[:, -3:])
        atom_types = torch.from_numpy(data[:, 0].astype(int)[:, None])
        one_hot = atom_types == self.atomic_number_list
        new_data["one_hot"] = one_hot
        if self.include_charges:
            new_data["charges"] = torch.zeros(n, 1, device=self.device)
        else:
            new_data["charges"] = torch.zeros(0, device=self.device)
        new_data["atom_mask"] = torch.ones(n, device=self.device)

        if self.sequential:
            edge_mask = torch.ones((n, n), device=self.device)
            edge_mask[~torch.eye(edge_mask.shape[0], dtype=torch.bool)] = 0
            new_data["edge_mask"] = edge_mask.flatten()
        return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=30,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h", action="store_true", help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default="~/diffusion/data/GEOM/")
    parser.add_argument("--data_file", type=str, default="drugs_crude.msgpack")
    args = parser.parse_args()
    extract_conformers(args)
