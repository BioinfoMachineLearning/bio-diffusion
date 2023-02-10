<div align="center">

# Bio-Diffusion

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Paper](http://img.shields.io/badge/arXiv-2302.04313-B31B1B.svg)](https://arxiv.org/abs/2302.04313)
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->
[![Datasets DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7542177.svg)](https://doi.org/10.5281/zenodo.7542177)
[![Checkpoints DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7626538.svg)](https://doi.org/10.5281/zenodo.7626538)

![Bio-Diffusion.png](./img/Bio-Diffusion.png)

</div>

## Description

A PyTorch hub of denoising diffusion probabilistic models designed to generate novel biological data

## How to run

Install Mamba

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Install dependencies

```bash
# clone project
git clone https://github.com/BioinfoMachineLearning/bio-diffusion
cd bio-diffusion

# create conda environment
mamba env create -f environment.yaml
conda activate bio-diffusion  # note: one still needs to use `conda` to (de)activate environments

# install local project as package
pip3 install -e .
```

Add or upgrade dependencies
```bash
# when installing a new package with pip or conda
# e.g., pip3 install .....

# update master configuration of environment layout
mamba env export | head -n -1 > environment.yaml

# also, be sure to remove the line `- bio-diffusion==0.0.1` from the list of `pip` dependencies generated

# push environment changes to remote for others to see
git add environment.yaml && git commit -m "Update Conda environment" && git push origin main

# then, others can update their local environments as follows
git pull origin main
mamba env update -f environment.yaml
```

Download data
```bash
# initialize data directory structure
mkdir -p data

# fetch, extract, and clean-up preprocessed data
cd data/
wget https://zenodo.org/record/7542177/files/EDM.tar.gz
tar -xzf EDM.tar.gz
rm EDM.tar.gz
cd ../
```

Download checkpoints

**Note**: Make sure to be located in the project's root directory beforehand (e.g., `~/bio-diffusion/`)
```bash
# fetch and extract model checkpoints directory
wget https://zenodo.org/record/7626538/files/GCDM_Checkpoints.tar.gz
tar -xzf GCDM_Checkpoints.tar.gz
rm GCDM_Checkpoints.tar.gz
```

## How to train

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

Train a model for small molecule generation with the QM9 dataset (**QM9**)

```bash
python3 src/train.py experiment=qm9_mol_gen_ddpm.yaml
```

**Note**: You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## How to evaluate

Reproduce our results for small molecule generation with the QM9 dataset

```bash
qm9_model_1_ckpt_path="checkpoints/QM9/Unconditional/model_1_epoch_499-EMA.ckpt"
qm9_model_2_ckpt_path="checkpoints/QM9/Unconditional/model_2_epoch_479-EMA.ckpt"
qm9_model_3_ckpt_path="checkpoints/QM9/Unconditional/model_3_epoch_459-EMA.ckpt"

python3 src/mol_gen_eval.py datamodule=edm_qm9 model=qm9_mol_gen_ddpm logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$qm9_model_1_ckpt_path" datamodule.dataloader_cfg.num_workers=1 model.diffusion_cfg.sample_during_training=false num_samples=10000 sampling_batch_size=100 num_test_passes=5

python3 src/mol_gen_eval.py datamodule=edm_qm9 model=qm9_mol_gen_ddpm logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$qm9_model_2_ckpt_path" datamodule.dataloader_cfg.num_workers=1 model.diffusion_cfg.sample_during_training=false num_samples=10000 sampling_batch_size=100 num_test_passes=5

python3 src/mol_gen_eval.py datamodule=edm_qm9 model=qm9_mol_gen_ddpm logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$qm9_model_3_ckpt_path" datamodule.dataloader_cfg.num_workers=1 model.diffusion_cfg.sample_during_training=false num_samples=10000 sampling_batch_size=100 num_test_passes=5
```

## How to sample
Generate small molecules similar to those contained within the QM9 dataset

```bash
qm9_model_ckpt_path="checkpoints/QM9/Unconditional/model_1_epoch_499-EMA.ckpt"

python3 src/mol_gen_sample.py datamodule=edm_qm9 model=qm9_mol_gen_ddpm logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$qm9_model_ckpt_path" num_samples=250 num_nodes=19 all_frags=true sanitize=false relax=false num_resamplings=1 jump_length=1 num_timesteps=1000 output_dir="./"
```

## Acknowledgements

Bio-Diffusion builds upon the source code and data from the following projects:

* [ClofNet](https://github.com/mouthful/ClofNet)
* [GBPNet](https://github.com/sarpaykent/GBPNet)
* [gvp-pytorch](https://github.com/drorlab/gvp-pytorch)
* [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
* [e3_diffusion_for_molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules)
* [DiffSBDD](https://github.com/arneschneuing/DiffSBDD)

We thank all their contributors and maintainers!
