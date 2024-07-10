# FROM nvcr.io/nvidia/pytorch:21.06-py3
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

LABEL authors="Colby T. Ford <colby@tuple.xyz>"

## Install system requirements
RUN apt update && \
    apt-get install -y --reinstall \
        ca-certificates && \
    apt install -y \
        git \
        vim \
        wget \
        libxml2 \
        libgl-dev \
        libgl1

## Set environment variables
ENV MPLCONFIGDIR /data/MPL_Config
ENV TORCH_HOME /data/Torch_Home
ENV TORCH_EXTENSIONS_DIR /data/Torch_Extensions
ENV DEBIAN_FRONTEND noninteractive

## Make directories
RUN mkdir -p /software/
WORKDIR /software/

## Install dependencies from Conda/Mamba
COPY environment.yaml /software/environment.yaml
RUN conda env create -f environment.yaml
RUN conda init bash && \
    echo "conda activate bio-diffusion" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

## Install bio-diffusion
RUN git clone https://github.com/BioinfoMachineLearning/bio-diffusion && \
    cd bio-diffusion && \
    pip install -e .
WORKDIR /software/bio-diffusion/

CMD /bin/bash