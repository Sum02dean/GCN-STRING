import os
import sys
from utilities.gcn_utills import *
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import networkx as nx
import glob
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import convert
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from utilities.gcn_utills import GraphMaker


class GraphMaker:
    """
    Main class to generate graphs
    """

    def __init__(self, anndata_path='./adata.csv'):
        self.anndata_path = anndata_path
        self.anndata = pd.read_csv(anndata_path, sep='\t')
        self.D = {}

    def collect_data(self, x):
        """Call as a lambda function on each row of a pandas data frame.
        Remeber to instantaite D = {} as as a global var in the main script.

        :param x: each row of a pandas.core.DataFrame
        :type x: list
        """

        # Feature lists
        dca_score = []
        pos_1 = []
        pos_2 = []

        # Grab name
        name_1 = x[0]
        name_2 = x[1]
        combined_name = "and".join([name_1, name_2])

        # Extract dca scores and positions
        for i in range(2, len(x), 3):
            dca_score.append(x[i])
            pos_1.append(x[i + 1])
            pos_2.append(x[i + 2])

        if combined_name not in self.D.values():
            # Add the data to the named entry
            self.D[combined_name] = {'dca': dca_score,
                                     'pos_1': pos_1, 'pos_2': pos_2}

            # Reset the feature lists
            dca_score = []
            pos_1 = []
            pos_2 = []

    def extract_data(self, df):
        """Modifies a globally defined dictionary with protein names,
        DCA values, and residue positions 

        :param df: Tao's new data
        :type df: pandas.core.DataFrame

        :return: There is not return type. Global D (dict) will be modified.
        :rtype: None
        """
        _ = df.apply(lambda x: self.collect_data(x), axis=1)
        return None

    def get_position_wise_df(self, x, protein_1, protein_2):

        # Intialise D dict
        _ = self.extract_data(x, )

        dca_dict = {}
        protein_pair = self.D["and".join([protein_1, protein_2])]

        # Get the top 20 dca stats, if none available set to zero
        pos_1 = protein_pair['pos_1']
        pos_2 = protein_pair['pos_2']
        dca = protein_pair['dca']
        dca_bridges = list(zip(pos_1, pos_2))

        # Populate the dictionary
        l = list(zip(pos_1, dca))
        for pos, score in l:
            if pos not in dca_dict.keys():
                dca_dict[pos] = [score]
            else:
                dca_dict[pos].append(score)

        # Take the max value
        dca_dict = {}
        for k, v in dca_dict.items():
            dca_dict[k] = max(v)

        df = pd.DataFrame({'pos_1': pos_1, 'pos_2': pos_2, 'dca': dca})

        return df, dca, dca_bridges

    def generate_alpha_fold_structures(self, string_to_af, pair_1, pair_2):
        """Queries alphs-fold predictions for a given protein sequnece and returns
         alpha-fld predicted structures.
        :param string_to_af: maping path between string-alphaFold
        :type string_to_af: string
        :param pair_1: protein name 1
        :type pair_1: string
        :param pair_2: protein name 2
        :type pair_2: string
        :return: alphaFold structures for each protein
        :rtype: objects
        """

        # Map from STRING to Alpha-Fold
        ecoli_maps = pd.read_csv(
            string_to_af, sep='\t', engine='python', header=None)
        map_1 = ecoli_maps[ecoli_maps[0] == pair_1]
        map_2 = ecoli_maps[ecoli_maps[0] == pair_2]

        # Save and import map files
        test_file_1 = map_1.iloc[0, -1]
        test_file_1 = test_file_1.replace('.gz', '')
        test_file_2 = map_2.iloc[0, -1]
        test_file_2 = test_file_2.replace('.gz', '')

        # Create sloppy parser
        sloppy_parser = PDBParser(
            structure_builder=SloppyStructureBuilder())
        protein_1 = sloppy_parser.get_structure(id=None, file=test_file_1)
        protein_2 = sloppy_parser.get_structure(id=None, file=test_file_2)

        # Structure 1
        sloppyio_1 = SloppyPDBIO()
        sloppyio_1.set_structure(protein_1)

        # Structure 2
        sloppyio_2 = SloppyPDBIO()
        sloppyio_2.set_structure(protein_2)

        # Get protein residue structures
        residues_1 = [x for x in sloppyio_1.structure.get_residues()]
        residues_2 = [x for x in sloppyio_2.structure.get_residues()]
        return residues_1, residues_2

    def calculate_residue_dist(self, seq_1, seq_2):
        """Calculates the euclidean distance between two residues in 3D space.
        :param residue_one: reference residue
        :type residue_one:  object
        :param residue_two: target residue
        :type residue_two: object
        :return: sqaured euclidean distance
        :rtype: float
        """
        diff_vector = seq_1["CA"].coord - seq_2["CA"].coord
        sq_dist = np.sqrt(np.sum(diff_vector * diff_vector))
        return sq_dist

    def calculate_dist_matrix(self, seq_1, seq_2):
        """Calculates the distance matrix for all pairwise residues
        :param seq_1: protein sequence 1
        :type seq_1: string
        :param seq_2: protein sequence 2
        :type seq_2: string
        :return: an nd array which encodes pairwise residue distances
        :rtype: np.array
        """
        d_mat = np.zeros((len(seq_1), len(seq_2)), np.float)
        for row, residue_one in enumerate(seq_1):
            for col, residue_two in enumerate(seq_2):
                euclidean_dist = self.calculate_residue_dist(
                    residue_one, residue_two)
                d_mat[row, col] = euclidean_dist
        return d_mat

    def generate_proximity_matrix(self, seq_1, seq_2, angstroms=10, show=False):
        """Creates an adacency matrix for points within n angstroms of each other
        :param seq_1: protein sequence 1
        :type seq_1: string
        :param seq_2: protein sequence 2
        :type seq_2: sting
        :param angstroms: max distance threshold , defaults to 10
        :type angstroms: int, optional
        :param show: to plot matrix, defaults to False
        :type show: bool, optional
        :return: a proximity matrix for points considered less than n angstroms apart
        :rtype: np.array
        """

        # Select the residues from maps that are less than 'n' angstoms apart
        contact_map = self.calculate_dist_matrix(seq_1, seq_2)
        adjacency_matrix = np.zeros(np.shape(contact_map))
        adjacency_matrix[contact_map < angstroms] = 1

        if show:
            plt.subplots(figsize=(12, 12))
            sns.heatmap(contact_map)
            plt.show()

            plt.subplots(figsize=(12, 12))
            sns.heatmap(adjacency_matrix)
            plt.show()

        return adjacency_matrix

    def generate_graphs(self, adjacency_matrix_1, adjacency_matrix_2, show=False):
        """Generates the initial graphs from provided adjacency matrices.
        :param adjacency_matrix_1: proximity matrix for protein 1
        :type adjacency_matrix_1: np.array
        :param adjacency_matrix_2: proximity matrix for protein 2
        :type adjacency_matrix_2: np.array
        :param show: plot graphs, defaults to False
        :type show: bool, optional
        :return: returns networkX graphs corrresponding to each adjacency matrix
        :rtype: tuple of networkX objects
        """
        # Generate graphs
        G_1 = nx.from_numpy_matrix(adjacency_matrix_1)
        G_2 = nx.from_numpy_matrix(adjacency_matrix_2)

        if show:
            # Plot the graphs
            plt.subplots(figsize=(12, 12))
            nx.draw(G_1, with_labels=False, edge_color="black",
                    node_size=10, width=0.2)
            plt.show()

            plt.subplots(figsize=(12, 12))
            nx.draw(G_2, with_labels=False, edge_color="black",
                    node_size=10, node_color='green', width=0.2)
            plt.show()
        return G_1, G_2

    def populate_graph_features(self, graph_1, graph_2, protein_1, protein_2, netsurf_path_dict):
        """ Populates each node (residue) with its respective net-surf feature vector

        :param graph_1: graph for protein 1
        :type graph_1: networkX graph object

        :param graph_2: graph for protein 2
        :type graph_2: networkX graph object

        :return: graph1 and graph2 populated with node features
        :rtype: tuple of networkX graphs
        """
        # Get netsurfp features for protein 1
        path_1 = netsurf_path_dict[protein_1]
        x_1 = pd.read_csv(path_1)

        # Get netsurfp features for protein 2
        path_2 = netsurf_path_dict[protein_2]
        x_2 = pd.read_csv(path_2)

        # Protein-1
        vars_to_keep = [x for x in x_1.columns if x not in [
            'id', 'seq', 'n', 'q3', 'q8']]
        features_p1 = x_1.loc[:, vars_to_keep]

        # Protein-2
        vars_to_keep = [x for x in x_2.columns if x not in [
            'id', 'seq', 'n', 'q3', 'q8']]
        features_p2 = x_2.loc[:, vars_to_keep]

        # Populate node features before making Union on graphs
        G_1_features = {}
        for i, node in enumerate(graph_1.nodes):
            feature_array = {'x': features_p1.iloc[i, :].values}
            G_1_features[node] = feature_array

        G_2_features = {}
        for i, node in enumerate(graph_2.nodes):
            feature_array = {'x': features_p2.iloc[i, :].values}
            G_2_features[node] = feature_array

        # Set the node attributes
        nx.set_node_attributes(graph_1, G_1_features)
        nx.set_node_attributes(graph_2, G_2_features)

        return graph_1, graph_2

    def link_graphs(self, graph_1, graph_2, dca_bridges, show=False):
        """Linkes the two protein graphs on their top 'n' DCA connections (bridges)
        :param graph_1: graph for protein 1
        :type graph_1: networkX graph object
        :param graph_2: graph for protein 2
        :type graph_2: networkX graph object
        :param dca_bridges: dca connections
        :type dca_bridges: list
        :param msa_coder_1: msa-seq coder for protein 1
        :type msa_coder_1: dict
        :param msa_coder_2: sa-seq coder for protein 2
        :type msa_coder_2: dict
        :param show: to plot graphs, defaults to False
        :type show: bool, optional
        :return: Union of the two protein graphs    
        :rtype: networkX graph object
        """

        # Connect the graphs together - use map-encoder for -b
        U = nx.union(graph_1, graph_2, rename=('a-', 'b-'))

        for (b1, b2) in dca_bridges:
            try:
                U.add_edge('a-' + str(b1),
                           'b-' + str(b2),
                           color='red')
            except Exception as e:
                print("something went wrong during graph-linking.")
                print(e)
                pass

        # Colour the DCA edges
        edge_color_list = []
        for (e1, e2) in U.edges:
            if e1[0] != e2[0]:
                edge_color_list.append('red')
            else:
                edge_color_list.append('black')

        node_color_list = []
        for node in U.nodes:
            if node[0] == 'a':
                node_color_list.append('blue')
            else:
                node_color_list.append('green')

        if show:
            f, ax = plt.subplots(figsize=(12, 12))
            nx.draw(U, edge_color=edge_color_list,
                    node_color=node_color_list, node_size=10, width=0.2)
            plt.show()
        return U

    def populate_edge_features(self, U, x, protein_1, protein_2):

        # make data into a dataframe
        df, _, _ = self.get_position_wise_df(x, protein_1, protein_2)

        for edge in U.edges:
            if ('a' in edge[0]) and ('b' in edge[1]):

                # Pull out the edges with DCA connection (max DCA)
                e_1 = int(edge[0].split('-')[-1])
                e_2 = int(edge[1].split('-')[-1])

                # Extract and join on the DCA scores
                dca_score = df[(df['pos_1'] == e_1) & (
                    df['pos_2'] == e_2)]['dca'].values

                # Set dca edge type to max dca score
                attrs = {(edge[0], edge[1]): {"dca": float(dca_score)}}
                nx.set_edge_attributes(U, attrs)

                # Set peptide edge type to zero
                attrs = {(edge[0], edge[1]): {"peptide": 0.0}}
                nx.set_edge_attributes(U, attrs)

            else:
                # Set dca edge type to 0
                attrs = {(edge[0], edge[1]): {"dca": 0.0}}
                nx.set_edge_attributes(U, attrs)

                # Set peptides edge type to 1
                attrs = {(edge[0], edge[1]): {"peptide": 1.0}}
                nx.set_edge_attributes(U, attrs)
        return U

    def save_graph_data(self, U, protein_1, protein_2, label):

        # Create graph_data dir if not present
        graph_name = "and".join([protein_1, protein_2])
        graph_folder_name = 'dca_graph_data'
        labels_folder_name = 'dca_graph_labels'

        # Graph dir
        isExist = os.path.exists(graph_folder_name)
        if not isExist:
            os.makedirs(graph_folder_name)
            print("{} directory created.".format(graph_folder_name))

        # Label dir
        isExist = os.path.exists(labels_folder_name)
        if not isExist:
            # Create it
            os.makedirs(labels_folder_name)
            print("{} directory created.".format(labels_folder_name))

        nx.write_gpickle(U, os.path.join(
            graph_folder_name, graph_name + ".gpickle"))

        # Create labels
        if label == 'P':
            bin_label = '1'
        else:
            bin_label = '0'

        # Format data to save on the fly
        labels_fn = os.path.join(labels_folder_name, 'labels.csv')
        fieldnames = ['protein_1', 'protein_2', 'label']
        row = {'protein_1': protein_1,
               'protein_2': protein_2, 'label': bin_label}

        # Open the file to append data to - only save new entries
        with open(labels_fn, 'a') as fd:
            writer = csv.DictWriter(fd, fieldnames=fieldnames)

            # Open file using seperate reader, and check the rows
            with open(labels_fn, 'r') as file1:
                existing_lines = [
                    line for line in csv.reader(file1, delimiter=',')]
                row_check = [x for x in row.values()]

                # If header already present, don't write
                if fieldnames not in existing_lines:
                    writer.writeheader()
                # If row already present, don't write
                if row_check not in existing_lines:
                    writer.writerow(row)


# Path to new data
root = '/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5/511145_EggNOGmaxLevel1224_eggNOGfilteredData/STRINGPhyBalancePhyla_Benchmark/'
meta = 'fixedNegVSposRratio_metadata.csv'
phyla = 'fixedNegVSposRratio_WithPhylaintegration_InSubjectOriPos_listDict_allPPs.csv'
no_phyla = 'fixedNegVSposRratio_WithoutPhylaintegration_InSubjectOriPos_listDict_allPPs.csv'


# Annotation file
anndata_path = ('/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5/'
                '511145_EggNOGmaxLevel1224_eggNOGfilteredData/STRINPhyPPI_Benchmark/allPPI_allInfo_frame.csv')
# Netsurf data
netsurf_path = "/mnt/mnemo6/tao/PPI_Coevolution/STRING_data_11.5/511145_netsurfp2_output/"

# ALPHA-Fold paths
string_to_af = "/mnt/mnemo6/damian/STRING_derived_v11.5/alphafold/mapping/83333.511145.tsv"
string_to_pdb = '/mnt/mnemo6/damian/STRING_derived_v11.5/pdb/pdb2string.blastp.best_score.tsv'
pdb_files_for_PDB = '/mnt/mnemo6/damian/STRING_freeze_v11.5/pdb/data/biounit/coordinates/divided/'


# Data
new_data = pd.read_csv(os.path.join(root, no_phyla), sep='\t', header=None)
phyla_data = pd.read_csv(os.path.join(root, phyla), sep='\t', header=None)
meta_data = pd.read_csv(os.path.join(root, meta), sep='\t', header=0)

# Netsurf outputs
netsurf = glob.glob(os.path.join(netsurf_path, "*.csv"))
netsurf.sort()

# Generate map between protein names and netsurfp file paths
seq_names = [x.split("/")[-1].replace(".csv", "") for x in netsurf]
netsurf_d = dict(zip(seq_names, netsurf))

# Load meta
meta_data = pd.read_csv(os.path.join(root, meta), sep='\t')

GM = GraphMaker(anndata_path=os.path.join(root, meta))
anndata = GM.anndata
rows, cols = np.shape(anndata)

# Loop over all instances in the anndata file
for i in range(rows):

    obs = anndata.iloc[i, :]
    pair_1 = obs['STRING_ID1']
    pair_2 = obs['STRING_ID2']
    label = obs['benchmark_status']
    print(obs)

    # Get alpha-fold structures
    residue_1, residue_2 = GM.generate_alpha_fold_structures(
        string_to_af, pair_1, pair_2)

    # Get proximity matrices
    pm_1 = GM.generate_proximity_matrix(
        seq_1=residue_1,
        seq_2=residue_1,
        angstroms=10,
        show=False)

    pm_2 = GM.generate_proximity_matrix(
        seq_1=residue_2,
        seq_2=residue_2,
        angstroms=10,
        show=False)

    # Generate graphs
    G_1, G_2 = GM.generate_graphs(
        adjacency_matrix_1=pm_1, adjacency_matrix_2=pm_2)

    # Populate graph attributes
    G_1, G_2 = GM.populate_graph_features(
        graph_1=G_1,
        graph_2=G_2,
        protein_1=pair_1,
        protein_2=pair_2,
        netsurf_path_dict=netsurf_d)

    # Get DCA bridge connections
    _, _, dca_bridges = GM.get_position_wise_df(
        x=new_data,
        protein_1=pair_1,
        protein_2=pair_2)

    # Link the graphs together on dca brides
    U = GM.link_graphs(G_1, G_2, dca_bridges=dca_bridges, show=False)

    # Populate edge attributes
    U = GM.populate_edge_features(
        U=U,
        x=new_data,
        protein_1=pair_1,
        protein_2=pair_2)

    # Save the graph to file
    GM.save_graph_data(U=U, protein_1=pair_1, protein_2=pair_2, label=label)
    break

print("Script finished without error.")