"""
From https://github.com/ehoogeboom/e3_diffusion_for_molecules/
"""

import numpy as np
import torch
import os
import logging

from typing import Dict, List
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from urllib.request import urlopen

from src.datamodules.components.edm.download import prepare_dataset
from src.datamodules.components.edm_dataset import ProcessedDataset

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


def download_data(url, outfile="", binary=False):
    """
    Downloads data from a URL and returns raw data.

    Parameters
    ----------
    url : str
        URL to get the data from
    outfile : str, optional
        Where to save the data.
    binary : bool, optional
        If true, writes data in binary.
    """
    # Try statement to catch downloads.
    try:
        # Download url using urlopen
        with urlopen(url) as f:
            data = f.read()
        logging.info("Data download success!")
        success = True
    except:
        logging.info("Data download failed!")
        success = False

    if binary:
        # If data is binary, use "wb" if outputting to file
        writeflag = "wb"
    else:
        # If data is string, convert to string and use "w" if outputting to file
        writeflag = "w"
        data = data.decode("utf-8")

    if outfile:
        logging.info("Saving downloaded data to file: {}".format(outfile))

        with open(outfile, writeflag) as f:
            f.write(data)

    return data, success


# Check if a string can be converted to an int, without throwing an error.
def is_int(str):
    try:
        int(str)
        return True
    except:
        return False

# Cleanup. Use try-except to avoid race condition.


def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass


def initialize_datasets(args, data_dir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False,
                        remove_h=False, create_pyg_graphs=False,
                        num_radials=1, device="cpu"):
    """
    Initialize datasets.

            included_species=all_species,
            subtract_thermo=subtract_thermo
    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    data_dir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "QM9" or "MD17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    remove_h: bool, optional
        If True, remove hydrogens from the dataset
    create_pyg_graphs: bool, optional
        If true, return PyTorch Geometric graphs when requesting dataset examples
    num_radials: int, optional
        Number of radial (distance) features to compute for each edge
    device: Union[torch.device, str], optional
        On which device to create graph features

    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {"train": args.num_train, "test": args.num_test, "valid": args.num_valid}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        data_dir, "QM9", subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}

    if dataset != "QM9":
        np.random.seed(42)
        fixed_perm = np.random.permutation(len(datasets["train"]["num_atoms"]))
        if dataset == "QM9_second_half":
            sliced_perm = fixed_perm[len(datasets["train"]["num_atoms"])//2:]
        elif dataset == "QM9_first_half":
            sliced_perm = fixed_perm[0:len(datasets["train"]["num_atoms"]) // 2]
        else:
            raise Exception("Wrong dataset name")
        for key in datasets["train"]:
            datasets["train"][key] = datasets["train"][key][sliced_perm]

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), "Datasets must have same set of keys!"

    # TODO: remove hydrogens here if needed
    if remove_h:
        for key, dataset in datasets.items():
            pos = dataset["positions"]
            charges = dataset["charges"]
            num_atoms = dataset["num_atoms"]

            # Check that charges corresponds to real atoms
            assert torch.sum(num_atoms != torch.sum(charges > 0, dim=1)) == 0

            mask = dataset["charges"] > 1
            new_positions = torch.zeros_like(pos)
            new_charges = torch.zeros_like(charges)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                p = pos[i][m]   # positions to keep
                p = p - torch.mean(p, dim=0)    # Center the new positions
                c = charges[i][m]   # Charges to keep
                n = torch.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i, :n] = c

            dataset["positions"] = new_positions
            dataset["charges"] = new_charges
            dataset["num_atoms"] = torch.sum(dataset["charges"] > 0, dim=1)

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    datasets = {
        split: ProcessedDataset(data,
                                num_pts=num_pts.get(split, -1),
                                included_species=all_species,
                                subtract_thermo=subtract_thermo,
                                create_pyg_graphs=create_pyg_graphs,
                                num_radials=num_radials,
                                device=device)
        for split, data in datasets.items()
    }

    # Check that all datasets have the same included species:
    assert(
        len(set(tuple(data.included_species.tolist()) for data in datasets.values())) == 1), \
        "All datasets must have same included_species! {}".format(
            {key: data.included_species for key, data in datasets.items()}
    )

    # These parameters are necessary to initialize the network
    num_species = datasets["train"].num_species
    max_charge = datasets["train"].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets["train"].num_pts
    args.num_valid = datasets["valid"].num_pts
    args.num_test = datasets["test"].num_pts

    return args, datasets, num_species, max_charge


def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels shoguld be integers.

    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset["charges"].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species["charges"].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                "The number of species is not the same in all datasets!")
        else:
            raise ValueError(
                "Not all datasets have the same number of species!")

    # Finally, return a list of all species
    return all_species


def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == "QM9":
        return compute_mean_mad_from_dataloader(dataloaders["train"], properties)
    elif dataset_name == "QM9_second_half" or dataset_name == "QM9_second_half":
        return compute_mean_mad_from_dataloader(dataloaders["valid"], properties)
    else:
        raise Exception("Wrong dataset name")


def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]["mean"] = mean
        property_norms[property_key]["mad"] = mad
    return property_norms


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


@typechecked
def prepare_context(
    conditioning: List[str],
    batch: Batch,
    property_norms: Dict[str, Dict[str, torch.Tensor]],
    positions_key: str = "x",
    atom_mask_key: str = "mask"
) -> TensorType["batch_num_nodes", "num_conditions"]:
    node_mask = batch[atom_mask_key].unsqueeze(-1)
    batch_size, num_nodes = (batch.index.shape[0], batch[positions_key].shape[0])

    context_list = []
    context_node_nf = 0
    for key in conditioning:
        properties = batch[key]
        properties = (properties - property_norms[key]["mean"]) / property_norms[key]["mad"]

        if (len(properties.shape) == 1) and (properties.shape[0] == batch_size):
            # duplicate respective global graph features to all nodes in the corresponding batch
            assert properties.shape == (batch_size,)

            datum_properties_list = []
            for property, datum in zip(properties, unbatch(batch.one_hot, batch.batch)):
                datum_num_nodes = datum.shape[0]
                datum_properties_list.append(property.view(1, 1).repeat(datum_num_nodes, 1))

            datum_properties = torch.cat(datum_properties_list, dim=0)
            context_list.append(datum_properties)
            context_node_nf += 1

        elif (len(properties.shape) == 1) and (properties.shape[0] == num_nodes):
            # aggregate node features
            assert properties.shape[0] == num_nodes

            context_key = properties.unsqueeze(-1)
            context_list.append(context_key)
            context_node_nf += context_key.shape[1]

        else:
            raise ValueError(f"Invalid properties tensor size of shape {properties.shape}.")

    if len(context_list) > 0:
        # concatenate context features
        context = torch.cat(context_list, dim=-1)
        # mask "missing" nodes
        context = context * node_mask
        assert context.shape[-1] == context_node_nf
    else:
        context = torch.zeros_like(node_mask)

    return context
