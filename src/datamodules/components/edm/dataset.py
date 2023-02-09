"""
From https://github.com/ehoogeboom/e3_diffusion_for_molecules/
"""

import torch
import os

import torch.nn as nn
import src.datamodules.components.edm.build_geom_dataset as build_geom_dataset

from omegaconf import DictConfig
from functools import partial
from typing import Optional

from src.datamodules.components.edm import download_dataset

from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch_geometric.loader.dataloader import DataLoader as PyGDataLoader

from src.datamodules.components.edm.collate import PreprocessQM9
from src.datamodules.components.edm.datasets_config import get_dataset_info
from src.datamodules.components.edm.utils import initialize_datasets
from src.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int):
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def retrieve_dataloaders(
    dataloader_cfg: DictConfig,
    esm_model: Optional[nn.Module] = None,
    esm_batch_converter: Optional[nn.Module] = None
):
    if "QM9" in dataloader_cfg.dataset:
        batch_size = dataloader_cfg.batch_size
        num_workers = dataloader_cfg.num_workers
        filter_n_atoms = dataloader_cfg.filter_n_atoms
        # Initialize dataloader
        cfg_, datasets, _, charge_scale = initialize_datasets(dataloader_cfg,
                                                              dataloader_cfg.data_dir,
                                                              dataloader_cfg.dataset,
                                                              subtract_thermo=dataloader_cfg.subtract_thermo,
                                                              force_download=dataloader_cfg.force_download,
                                                              remove_h=dataloader_cfg.remove_h,
                                                              create_pyg_graphs=dataloader_cfg.create_pyg_graphs,
                                                              num_radials=dataloader_cfg.num_radials,
                                                              device=dataloader_cfg.device)
        qm9_to_eV = {
            "U0": 27.2114, "U": 27.2114, "G": 27.2114, "H": 27.2114,
            "zpve": 27211.4, "gap": 27.2114, "homo": 27.2114, "lumo": 27.2114
        }

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            log.info("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=dataloader_cfg.include_charges)
        dataloader_class = (
            partial(PyGDataLoader, prefetch_factor=100, worker_init_fn=set_worker_sharing_strategy)
            if dataloader_cfg.create_pyg_graphs
            else partial(TorchDataLoader, collate_fn=preprocess.collate_fn)
        )
        dataloaders = {split: dataloader_class(dataset,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               shuffle=cfg_.shuffle if (split == "train") else False,
                                               drop_last=cfg_.drop_last if (split == "train") else False)
                       for split, dataset in datasets.items()}
    elif "GEOM" in dataloader_cfg.dataset:
        data_file = os.path.join("data", "EDM", "GEOM", "GEOM_drugs_30.npy")
        dataset_info = get_dataset_info(dataloader_cfg.dataset, dataloader_cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=dataloader_cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          dataloader_cfg.include_charges,
                                                          dataloader_cfg.device,
                                                          dataloader_cfg.sequential)
        dataloaders = {}
        for split, data_list in zip(["train", "valid", "test"], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform,
                                                          create_pyg_graphs=dataloader_cfg.create_pyg_graphs,
                                                          num_radials=dataloader_cfg.num_radials,
                                                          device=dataloader_cfg.device)
            shuffle = (split == "train") and not dataloader_cfg.sequential

            # Sequential dataloading disabled for now.
            dataloader_class = (
                partial(build_geom_dataset.GeomDrugsPyGDataLoader, sequential=dataloader_cfg.sequential)
                if dataloader_cfg.create_pyg_graphs
                else partial(build_geom_dataset.GeomDrugsTorchDataLoader, sequential=dataloader_cfg.sequential)
            )
            dataloaders[split] = dataloader_class(
                dataset=dataset,
                batch_size=dataloader_cfg.batch_size,
                shuffle=shuffle
            )
        del split_data
        charge_scale = None
    else:
        raise ValueError(f"Unknown dataset {dataloader_cfg.dataset}")

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data["num_atoms"] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data["one_hot"].size(0)
        datasets[key].perm = None
    return datasets
