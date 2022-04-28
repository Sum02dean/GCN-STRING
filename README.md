![STRING logo](src/configs/logo.png)


# Predict Phyiscal Protein Interactions using Graph Neural Networks
This repository contains code predicting physical interaction of proteins using direct coupling analysis and graph neural networks. 

## Setup:
This code was designed to run on the [s3it](https://apps.s3it.uzh.ch/) cluster to avoid out of memory (OOM) issues.

## GPU Usage:
Because of the large datasets involved data-loading and training can take a large amount of time.
To run on s3it GPU cluster, make sure to have a GPU compatible version of tensorflow installed before running the code. Please visit [s3it-GPU instructions](https://docs.s3it.uzh.ch/cluster/python_tensorflow_example/).

### Installation: 
This requires a valid installation of [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Create the Python environment as described below:

```sh
cd configs
conda env create -f env.yml 
conda activate gcn_env
```
### Data:
You can find the data paths within the gcn_generator.py file (see scripts directory).

### Data Preparation:
To generate the  inter protein graphs, run:

```sh
cd src/scripts
bash run_graph_generator.sh
```
### Running the Model:
To run the model simply run:
```sh
cd src/scripts
bash run_graph_prediction.sh
```

### Model:

| Models | Resources |
| ------ | ------ | 
| Neural Net (Spektral GCN) | [Spektral model](https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/general_gnn.py) |

### Potential Conflicts:
There may be issues running this code from a local machine. It was designed to run on the s3it
cluster. 
### Major Dependencies:
The conda environemnt provided should contain all of these requirements. If not, you can find them at the following sources.

| Dependency | Installation |
| ------ | ------ | 
| Spektral (cpu, linux) |[Pypi](https://pypi.org/project/spektral/)|

