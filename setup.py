#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="Bio-Diffusion",
    version="0.0.1",
    description="A hub for deep diffusion networks designed to generate novel biological data",
    author="Alex Morehead",
    author_email="acmwhb@umsystem.edu",
    url="https://github.com/BioinfoMachineLearning/bio-diffusion",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
