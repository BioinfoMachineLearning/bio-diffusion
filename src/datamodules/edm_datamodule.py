# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import os

from typing import Any, Dict, Optional
from omegaconf import DictConfig

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.components.edm.dataset import retrieve_dataloaders


class EDMDataModule(LightningDataModule):
    """
    A data wrapper for the EDM datasets. It downloads any missing
    data files from Springer Nature or Zenodo.

    :param dataloader_cfg: configuration arguments for EDM dataloaders.
    """

    def __init__(self, dataloader_cfg: DictConfig):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataloader_train: Optional[DataLoader] = None
        self.dataloader_val: Optional[DataLoader] = None
        self.dataloader_test: Optional[DataLoader] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (e.g., self.x = y).
        """
        data_path = os.path.join(self.hparams.dataloader_cfg.data_dir, self.hparams.dataloader_cfg.dataset)

        if "QM9" in self.hparams.dataloader_cfg.dataset and not all([
            os.path.exists(os.path.join(data_path, "train.npz")),
            os.path.exists(os.path.join(data_path, "valid.npz")),
            os.path.exists(os.path.join(data_path, "test.npz"))
        ]):
            retrieve_dataloaders(self.hparams.dataloader_cfg)

        elif "GEOM" in self.hparams.dataloader_cfg.dataset and not all([
            os.path.exists(os.path.join(data_path, "GEOM_drugs_30.npy")),
            os.path.exists(os.path.join(data_path, "GEOM_drugs_n_30.npy")),
            os.path.exists(os.path.join(data_path, "GEOM_drugs_smiles.txt"))
        ]):
            retrieve_dataloaders(self.hparams.dataloader_cfg)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        Note: This method is called by Lightning with both `trainer.fit()` and `trainer.test()`.
        """
        # load dataloaders only if not loaded already
        if not self.dataloader_train and not self.dataloader_val and not self.dataloader_test:
            self.dataloaders, self.charge_scale = retrieve_dataloaders(
                self.hparams.dataloader_cfg,
                esm_model=getattr(self, "esm_model", None),
                esm_batch_converter=getattr(self, "esm_batch_converter", None)
            )
            self.dataloader_train, self.dataloader_val, self.dataloader_test = (
                self.dataloaders["train"], self.dataloaders["valid"], self.dataloaders["test"]
            )

    def train_dataloader(self):
        return self.dataloader_train

    def val_dataloader(self):
        return self.dataloader_val

    def test_dataloader(self):
        return self.dataloader_test

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "edm_qm9.yaml")
    cfg.data_dir = str(root / "data" / "EDM")
    _ = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "edm_geom.yaml")
    cfg.data_dir = str(root / "data" / "EDM")
    _ = hydra.utils.instantiate(cfg)

