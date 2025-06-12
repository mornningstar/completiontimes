# Completion Times
This repository contains research code for predicting file completion times in open source projects. It was developed 
as part of a master's thesis by Nastasja Stephanie Parschew.

The project fetches commit history from GitHub, stores it in a MongoDB database and engineers various 
features from the collected data. Several regression models are trained to estimate how long a file will take to be 
completed based on its history and metadata.
## Repository layout

- **src/** – implementation of data collection, feature engineering and modelling
- **scripts/** – utility scripts for database migrations and repairs
- **config/** – configuration files with project definitions and credentials
- **tests/** – unit tests for some of the components
- **environment.yml** – conda environment specification with all required dependencies

## Installation

Create a conda environment using the provided environment file:

```bash
conda env create -f environment.yml
conda activate rapids-25.02
```

The configuration in `config/config.py` requires a GitHub access token for downloading repository data. 
Replace the placeholder token with your own before running the pipeline.

## Running the pipeline

The main entry point is `src/main.py`. It processes the repositories defined in `config/projects.py`, performs feature 
engineering and trains the configured models:

```bash
python src/main.py
```

Results (trained models and visualisations) are written to timestamped folders under `runs/`.