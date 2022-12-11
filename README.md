# GCN DockingScalability

## Build an environment
### Docker
1. Create the Docker image `docker build -t your_image_name .`
2. Run your image `docker run -it -d --shm-size 128g your_image_name`
3. Access to your container `docker exec -it <container id> bash`

### Conda

1. Create the conda env `conda env create -f environment_gpu.yml`
2. Activate the env `conda activate conda_env_GCN_SCAL`
3. Install Pyg-Lib :
> pyg-lib provides efficient GPU-based routines to parallelize workloads in heterogeneous graphs across different node types and edge types.

`pip install pyg-lib -f https://data.pyg.org/whl/torch-1.12.0+cu113.html`

## Build the Graphs
1. Download the PDBBind database from http://www.pdbbind.org.cn/ with `scripts/download_pdbbind.sh`. Extracted PDBBind complexes are stored in *data/raw/*
2. Create the graphs `python data.py`

## Launch the trainings
1. You can set the `exp/commands_list.txt` to choose which trainings will be done.
2. Set `launch_training.sh` (lines 2 and 3) with the correct numbers
3. Run `./launch_training.sh`
