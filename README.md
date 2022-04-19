![STRING logo](configs/logo.jpg)


# Predict Phyiscal Protein Interactions
This repository contains code predicting physical interaction of proetins using directed coupling analysis and graph nerual netowkrs. 

### Installation 
This requires a valid installation of [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Create the Python environment as described below:

```sh
cd configs
conda env create -f env.yml 
conda activate gcn_env
```

### Data
You can find the data paths in the gcn_generator.py under the scripts directory, 

### Data Preparation
To generate the protein-protein graphs, run:
```sh
cd src/scripts
bash run_graph_generator.sh
```
### Running the model
To run the model simply navigate run:
```sh
cd src/scripts
bash run_graph_prediction.sh
```

### Model

| Models | Resources |
| ------ | ------ | 
| Neural Net (Spektral GCN) | [Spektral model](https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/general_gnn.py) |

### Potential conflicts
```py
"""Pandas Multiindex deprecation: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead."""
```
### Major dependencies
The conda environemnt provided should contain all of these requirements. If not, you can find them at the following sources.

| Dependency | Installation |
| ------ | ------ | 
| Spektral (cpu, linux) |[Pypi](https://pypi.org/project/spektral/)|
| Pytorch (cpu, linux) |[Pypi](https://pytorch.org/)|
| torch-summary |[Pypi](https://pypi.org/project/torch-summary/)|

