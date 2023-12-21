#!/bin/bash

set -e  # Stop on error

apt-get update
apt-get install -y tree

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install Python packages
pip install --upgrade pip
pip install kedro pyarrow fastparquet openpyxl kedro-docker

# Navigate to pipeline directory and install requirements
cd pipeline
pip install -r requirements.txt

# Initialize and build kedro docker
kedro docker init
kedro docker build
