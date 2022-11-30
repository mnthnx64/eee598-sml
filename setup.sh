#!/bin/sh
# This script is used to setup the environment for the
# project. It should be run from the root of the project

# Download the data
echo "Downloading data..."
mkdir -p dataset
wget -O dataset/data.csv https://www.dropbox.com/s/ncocaic8ej8mrw4/dataset.csv?dl=0

# Install the requirements with conda
echo "Installing requirements..."
conda env create -f environment.yml

# Activate the environment
echo "Activating environment..."
# source activate sml598

# Run the data processing script
echo "Processing data..."
python utils/process_data.py

