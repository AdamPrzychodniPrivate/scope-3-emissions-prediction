#!/bin/bash

# Create a directory for Miniconda installation
mkdir -p ~/miniconda3

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Cleanup installation script
rm -rf ~/miniconda3/miniconda.sh

# Initialize Conda
~/miniconda3/bin/conda init bash

# Create the Conda environment
~/miniconda3/bin/conda create --name kedro-environment python=3.10 -y

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kedro-environment

# Install dependencies from requirements.txt and environment.yml
pip install -r src/requirements.txt
