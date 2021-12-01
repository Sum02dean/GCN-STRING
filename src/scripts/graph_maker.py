import warnings
import csv
from tqdm import tqdm
import xpdb
from Bio.PDB import PDBParser
import networkx as nx
from numpy.linalg import matrix_rank
from matplotlib.backends.backend_pdf import PdfPages
from src.utilities.parallelize_dca_stats import parallelize_dataframe
from src.utilities.get_dca_stats import get_dca_stats, get_dca_stats_with_figs
from src.utilities.dca_lab import DcaLab
from src.utilities.frequency_coupler import FrequencyCoupler as fc
from src.utilities.msa_synthesizer import MsaSynthesizer as MSA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
sys.path.append("../")  # nopep8
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ALPHA-Fold paths
ecoli_ext_string_to_af = "/mnt/mnemo6/damian/STRING_derived_v11.5/alphafold/mapping/83333.511145.tsv"
string_to_pdb = '/mnt/mnemo6/damian/STRING_derived_v11.5/pdb/pdb2string.blastp.best_score.tsv'
pdb_files_for_PDB = '/mnt/mnemo6/damian/STRING_freeze_v11.5/pdb/data/biounit/coordinates/divided/'


# Tao STRING 11.5 paths
anndata_path = ('/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5/'
                '511145_EggNOGmaxLevel1224_eggNOGfilteredData/STRINPhyPPI_Benchmark/allPPI_allInfo_frame.csv')
paired_msa_path = ('/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5'
                   '/511145_EggNOGmaxLevel1224_eggNOGfilteredData/pair_MSA_Nf90_PasteAlign')
configs_dir = 'configs_11.5'
coevo_path = ('/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5'
              '/511145_EggNOGmaxLevel1224_eggNOGfilteredData/coevolutoin_result_DCA/')


class Main:
    """
    Main class to generate graphs
    """

    def __init__(self, anndata_path='./adata.csv'):
        self.anndata_path = anndata_path
        self.anndata = pd.read_csv(anndata_path, sep='\t')

    def __get_top_ranking_bet_value_dict(self, record=None, top_num=20):
        """This is a function written by Tao to break down the DCA matrix and fetch the
         top DCA-scoring pairs and their associated DCA scores"""

        query_pro1, query_pro2, l1, l2, coevo_path, suffix = record
        data_file_name = coevo_path+query_pro1+"and"+query_pro2+suffix+".npz"
        data_array_dig = np.load(data_file_name)['arr_0']
        data_array = data_array_dig.T + data_array_dig
        np.fill_diagonal(data_array, 0)
        bet_data_array = data_array[:l1, l1:]
        ascending_bet_data_array = np.sort(bet_data_array.flatten())
        descending_bet_data_array = ascending_bet_data_array[::-1]
        return_list = [query_pro1, query_pro2]
        esisted_top_values = set()

        scores = []
        pairs_1 = []
        pairs_2 = []

        for j in range(top_num):
            top_value = descending_bet_data_array[j]
            if top_value not in esisted_top_values:
                top_idx = np.where(bet_data_array == top_value)
                for i in range(len(top_idx[0])):
                    top_row, top_col = top_idx[0][i], top_idx[1][i]
                    top_col += l1
                    if len(return_list) < (2+top_num*3):
                        return_list.extend((top_value, top_row, top_col))
                        scores.append(top_value)
                        pairs_1.append(top_row)
                        pairs_2.append(top_col)

                esisted_top_values.add(top_value)
            if len(return_list) > (2+top_num*3):
                break

        df = pd.DataFrame(
            {'pair_1': pairs_1, 'pair_2': pairs_2, 'scores': scores})
        return df, (return_list), query_pro1, query_pro1

    def generate_top_dca_scores(self, pair_1, pair_2, len_1, len_2, coevo_path, n_dca=20,
                                fn_suffix='_pydcaFNAPC_array'):
        """Extracts the top _dca scores between two protein pairs

        :param pair_1: name of protein 1
        :type pair_1: string
        :param pair_2: name of protein 2
        :type pair_2: string
        :param len_1: length of protein 1
        :type len_1: int
        :param len_2: length of protein 2
        :type len_2: int
        :param coevo_path: path to coevolutionary data
        :type coevo_path: string
        :param n_dca: number of top dca scores to include, defaults to 20
        :type n_dca: int, optional
        :param fn_suffix: suffix of the coevolutionary file, defaults to '_pydcaFNAPC_array'
        :type fn_suffix: str, optional
        :return: top-10 dca for a given protein pair
        :rtype: list
        """
        record = [pair_1, pair_2, len_1, len_2, coevo_path, fn_suffix]
        dca_stats, _, _, _ = self.__get_top_ranking_bet_value_dict(
            record, top_num=n_dca)
        return dca_stats

    def generate_dca_bridges(self, coevo_path, pair_1, pair_2, len_1, len_2, n_dca=20):
        """Extracts the residue names corresponding to the top dca scores. These are used to connect the two
        protein graphs together on "bridges".

        :param coevo_path: path to coevolutionary data
        :type coevo_path: string
        :param pair_1: protein pair 1
        :type pair_1: string
        :param pair_2: protein pair 2
        :type pair_2: string
        :param len_1: protein length 1
        :type len_1: int
        :param len_2: protein length 2
        :type len_2: int
        :param n_dca: number of top dcas, defaults to 20
        :type n_dca: int, optional
        :return: list containing the tuple of sorted top ranking pairs
        :rtype: list
        """
        dca_stats = self.generate_top_dca_scores(pair_1=pair_1,
                                                 pair_2=pair_2,
                                                 len_1=len_1,
                                                 len_2=len_2,
                                                 n_dca=n_dca,
                                                 coevo_path=coevo_path)

        # Subset the residues
        res_1 = [int(x) for x in dca_stats.values[:, 0]]
        res_2 = [int(x) for x in dca_stats.values[:, 1]]

        # Get top 20-dca pairs, generate bridges
        res_a = res_1[0:n_dca]
        res_b = res_2[0:n_dca]
        scores = dca_stats['scores'].iloc[:n_dca]
        bridges = list(zip(res_a, res_b))
        return bridges

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
            structure_builder=xpdb.SloppyStructureBuilder())
        protein_1 = sloppy_parser.get_structure(id=None, file=test_file_1)
        protein_2 = sloppy_parser.get_structure(id=None, file=test_file_2)

        # Structure 1
        sloppyio_1 = xpdb.SloppyPDBIO()
        sloppyio_1.set_structure(protein_1)
        sloppyio_1.save("sloppyio_1.pdb")

        # Structure 2
        sloppyio_2 = xpdb.SloppyPDBIO()
        sloppyio_2.set_structure(protein_2)
        sloppyio_2.save("sloppyio_2.pdb")

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

    def map_net_surf_to_graphs(self, graph_2, pair_1, pair_2, ns_path_msa, ns_path_seq, suffix='_map2MSA.csv'):
        """Maps the graphs node labels to the corresponding points from MSA, and returns net-surf residue 
        features at these points.

        :param graph_2: graph for protein 2
        :type graph_2: networkX object
        :param pair_1: protein 1 name
        :type pair_1: string
        :param pair_2: protein 2 name
        :type pair_2: string
        :param ns_path_msa: net-surf path
        :type ns_path_msa: string
        :param ns_path_seq: net-surf path for protein sequence
        :type ns_path_seq: string
        :param suffix: file ending for MSA, defaults to '_map2MSA.csv'
        :type suffix: str, optional
        :return: mapping between msa to sequence, new node labels, and net-surf features for both proteins.
        :rtype: tuple
        """

        # -------- MSA-Seq Mapping ----------

        # Get the seq-MSA mapped feature matrix
        net = pair_1 + suffix
        fn = os.path.join(ns_path_msa, net)
        net_surf_1 = pd.read_csv(fn)

        # Get the seq-MSA mapped feature matrix
        net = pair_2 + suffix
        fn = os.path.join(ns_path_msa, net)
        net_surf_2 = pd.read_csv(fn)

        # -------- SEQUENCE ----------
        net = pair_1 + '.csv'
        fn = os.path.join(ns_path_seq, net)
        x_net_surf_1 = pd.read_csv(fn)

        net = pair_2 + '.csv'
        fn = os.path.join(ns_path_seq, net)
        x_net_surf_2 = pd.read_csv(fn)

        # Get new graph labels
        graph_1_labels = list(set([x for x in net_surf_1.seqPos.values]))

        # sequence and MSA maps for protein-1
        seq_pos_p1 = net_surf_1.seqPos
        msa_pos_p1 = net_surf_1.MSAPos
        msa_2_seq_p1 = dict(zip(msa_pos_p1, seq_pos_p1))
        seq_2_msa_p1 = dict(zip(seq_pos_p1, msa_pos_p1))

        # sequence and MSA maps for protein-2
        seq_pos_p2 = [x+(seq_pos_p1.values[-1]+1) for x in net_surf_2.seqPos]
        msa_pos_p2 = [x+(msa_pos_p1.values[-1]+1) for x in net_surf_2.MSAPos]
        msa_2_seq_p2 = dict(zip(msa_pos_p2, seq_pos_p2))
        seq_2_msa_p2 = dict(zip(seq_pos_p2, msa_pos_p2))

        # Reset G2 labels
        g2_labels = [x+(graph_1_labels[-1]) for x in x_net_surf_2.n.values]
        G_2_labels = dict(zip(graph_2.nodes, g2_labels))

        return msa_2_seq_p1, msa_2_seq_p2, G_2_labels, x_net_surf_1, x_net_surf_2

    def relabel_graph_nodes(self, graph, new_labels):
        """Overwrites node labels in a graph with new_labels.

        :param graph: networkX graph object
        :type graph: object
        :param new_labels: new labels to provide
        :type new_labels: list
        :return: graph object with renamed node labels
        :rtype: networkx object
        """
        # Relabel nodes
        new_graph = nx.relabel_nodes(graph, new_labels)
        return new_graph

    def check_bridge_connections(self, msa_coder_1, msa_coder_2, graph_1, graph_2, bridges):
        """Test if all bridge nodes are contained within their respective graphs - should be! 

        :param msa_coder_1: msa-seq coder for protein 1
        :type msa_coder_1: dict
        :param msa_coder_2: msa-seq coder for protein 2
        :type msa_coder_2: dict
        :param graph_1: networkx graph 1    
        :type graph_1: object
        :param graph_2: networkX graph 2    
        :type graph_2: object
        :param bridges: list of dca bridges
        :type bridges: list
        :return: 0 if tests passed, else 1
        :rtype: int
        """

        g1_e = set(sorted([x[1] for x in graph_1.edges]))
        g2_e = [x for x in set(sorted([x[1] for x in graph_2.edges]))]

        for i, (b1, b2) in enumerate(bridges):
            if (msa_coder_1[b1] not in g1_e):
                print('failed with exit status: 1')
                return 1

            if (msa_coder_2[b2] not in g2_e):
                print('failed with exit status: 1')
                return 1
        return 0

    def populate_graph_features(self, graph_1, graph_2, x_net_surf_1, x_net_surf_2):
        """ Populates each node (residue) with its respective net-surf feature vector

        :param graph_1: graph for protein 1
        :type graph_1: networkX graph object
        :param graph_2: graph for protein 2
        :type graph_2: networkX graph object
        :param x_net_surf_1: net-surf feautures for protein 1
        :type x_net_surf_1: pandas dataframe
        :param x_net_surf_2: net-surf features for protein 2
        :type x_net_surf_2: pandas dataframe
        :return: graph1 and graph2 populated with node features
        :rtype: tuple of networkX graphs
        """

        # Protein-1
        vars_to_keep = [x for x in x_net_surf_1.columns if x not in [
            'id', 'seq', 'n', 'q3', 'q8']]
        features_p1 = x_net_surf_1.loc[:, vars_to_keep]

        # Protein-2
        vars_to_keep = [x for x in x_net_surf_2.columns if x not in [
            'id', 'seq', 'n', 'q3', 'q8']]
        features_p2 = x_net_surf_2.loc[:, vars_to_keep]

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

    def link_graphs(self, graph_1, graph_2, bridges, msa_coder_1, msa_coder_2, show=False):
        """Linkes the two protein graphs on their top 'n' DCA connections (bridges)

        :param graph_1: graph for protein 1
        :type graph_1: networkX graph object
        :param graph_2: graph for protein 2
        :type graph_2: networkX graph object
        :param bridges: dca connections
        :type bridges: list
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

        for (b1, b2) in bridges:
            try:
                U.add_edge('a-' + str(msa_coder_1[b1]),
                           'b-' + str(msa_coder_2[b2]),
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

    def get_computational_graph(self, string_to_af, ns_path_msa, ns_path_seq,
                                coevo_path, n_samples=2, n_dca=20, show=False):
        """High level method to generate graphs from annotation file on all proteins.

        :param string_to_af: path that maps STRING to AlphaFold
        :type string_to_af: string
        :param ns_path_msa: net-surf path for MSA
        :type ns_path_msa: string
        :param ns_path_seq: net-surf path for protein seq
        :type ns_path_seq: string   
        :param coevo_path: path to co-evolution data
        :type coevo_path: string
        :param n_samples: number of graphs to produce, defaults to 2
        :type n_samples: int, optional
        :param n_dca: number of top ranked dca scores to consder, defaults to 20
        :type n_dca: int, optional
        :param show: plot all , defaults to False
        :type show: bool, optional
        :return: tuple containing list of graphs and list of labels
        :rtype: tuple
        """

        # 1. Loop over all observations in anndata file
        graphs = []
        labels = []
        for (pair_1, pair_2) in tqdm(self.anndata.iloc[:n_samples, 0:2].values):
            try:
                len_1 = int(df.loc[(df['STRING_ID1'] == pair_1) & (
                    df['STRING_ID2'] == pair_2)].len1)
                len_2 = int(df.loc[(df['STRING_ID1'] == pair_1) & (
                    df['STRING_ID2'] == pair_2)].len2)
                label = df.loc[(df['STRING_ID1'] == pair_1) & (
                    df['STRING_ID2'] == pair_2)].benchmark_status.values
                labels.append(label)

                # 2. Generate_dca_bridges(self, n=20)
                bridges = self.generate_dca_bridges(pair_1=pair_1,
                                                    pair_2=pair_2,
                                                    len_1=len_1,
                                                    len_2=len_2,
                                                    n_dca=n_dca,
                                                    coevo_path=coevo_path)
                # 3. Generate alpha-fold residues
                residue_1, residue_2 = self.generate_alpha_fold_structures(string_to_af,
                                                                           pair_1=pair_1,
                                                                           pair_2=pair_2)

                # 4. Generate proximity adjacency matrices
                am_1 = self.generate_proximity_matrix(
                    seq_1=residue_1, seq_2=residue_1, angstroms=10, show=show)
                am_2 = self.generate_proximity_matrix(
                    seq_1=residue_2, seq_2=residue_2, angstroms=10, show=show)

                # 5. Build graph
                G_1, G_2 = self.generate_graphs(am_1, am_2, show=show)

                # 6. Parse net_surf
                outputs = self.map_net_surf_to_graphs(ns_path_msa=ns_path_msa,
                                                      ns_path_seq=ns_path_seq,
                                                      pair_1=pair_1,
                                                      pair_2=pair_2,
                                                      graph_2=G_2,
                                                      suffix='_map2MSA.csv')

                # 7. Collect outputs
                msa_2_seq_p1, msa_2_seq_p2, G_2_labels, x_net_surf_1, x_net_surf_2 = outputs

                # 8. Re-label graph
                G_2 = self.relabel_graph_nodes(graph=G_2,
                                               new_labels=G_2_labels)

                # 9. Check bridge stability
                self.check_bridge_connections(msa_coder_1=msa_2_seq_p1,
                                              msa_coder_2=msa_2_seq_p2,
                                              graph_1=G_1,
                                              graph_2=G_2,
                                              bridges=bridges)

                # 10. Populate features
                G_1, G_2 = self.populate_graph_features(graph_1=G_1,
                                                        graph_2=G_2,
                                                        x_net_surf_1=x_net_surf_1,
                                                        x_net_surf_2=x_net_surf_2)

                # 11. Link graphs
                U = self.link_graphs(graph_1=G_1,
                                     graph_2=G_2,
                                     bridges=bridges,
                                     msa_coder_1=msa_2_seq_p1,
                                     msa_coder_2=msa_2_seq_p2,
                                     show=show)

                # 12. Save and return graphs
                graphs.append(U)
                graph_name = "and".join([pair_1, pair_2])
                nx.write_gpickle(U, os.path.join(
                    'graph_data', graph_name + ".gpickle"))

                # Create labels
                if label == 'P':
                    bin_label = '1'
                else:
                    bin_label = '0'

                # Format data to save on the fly
                labels_fn = os.path.join('graph_labels', 'labels.csv')
                fieldnames = ['protein_1', 'protein_2', 'label']
                row = {'protein_1': pair_1,
                       'protein_2': pair_2, 'label': bin_label}

                # Open the file to append data to - only save new entries
                with open(labels_fn, 'a') as fd:
                    writer = csv.DictWriter(fd, fieldnames=fieldnames)

                    # Open file using seperate reader, and check the rows
                    with open('graph_labels/labels.csv', 'r') as file1:
                        existing_lines = [
                            line for line in csv.reader(file1, delimiter=',')]
                        row_check = [x for x in row.values()]

                        # If header already present, don't write
                        if fieldnames not in existing_lines:
                            writer.writeheader()
                        # If row already present, don't write
                        if row_check not in existing_lines:
                            writer.writerow(row)

            # If something breaks - pass
            except Exception as e:
                print('Skipped')
                print(e)
                pass

        return graphs, labels


# ALPHA-Fold Paths
ecoli_ext_string_to_af = "/mnt/mnemo6/damian/STRING_derived_v11.5/alphafold/mapping/83333.511145.tsv"
string_to_pdb = '/mnt/mnemo6/damian/STRING_derived_v11.5/pdb/pdb2string.blastp.best_score.tsv'
pdb_files_for_PDB = '/mnt/mnemo6/damian/STRING_freeze_v11.5/pdb/data/biounit/coordinates/divided/'

# Tao STRING 11.5 paths
anndata_path = '/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5/511145_EggNOGmaxLevel1224_eggNOGfilteredData/STRINPhyPPI_Benchmark/allPPI_allInfo_frame.csv'
paired_msa_path = '/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5/511145_EggNOGmaxLevel1224_eggNOGfilteredData/pair_MSA_Nf90_PasteAlign'
configs_dir = 'configs_11.5'
coevo_path = '/mnt/mnemo6/tao/PPI_Coevolution/CoEvo_data_STRING11.5/511145_EggNOGmaxLevel1224_eggNOGfilteredData/coevolutoin_result_DCA/'

# Netsurf paths
nsp_msa = "/mnt/mnemo6/tao/PPI_Coevolution/STRING_data_11.5/511145_netsurfp2_output/_EggNOGmaxLevel1224_map2MSA"
nsp_seq = "/mnt/mnemo6/tao/PPI_Coevolution/STRING_data_11.5/511145_netsurfp2_output/"


df = pd.read_csv(anndata_path, sep='\t')
# n_samples = use -1 for all
n_samples = -1
P = Main(anndata_path=anndata_path)
graphs, labels = P.get_computational_graph(n_samples=n_samples, string_to_af=ecoli_ext_string_to_af,
                                           ns_path_msa=nsp_msa,
                                           ns_path_seq=nsp_seq,
                                           coevo_path=coevo_path,
                                           show=False)
