import pickle
import os
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import networkx as nx
import pandas as pd
import csv
import matplotlib.pyplot as plt
import copy
import Bio.PDB
import Bio.PDB.StructureBuilder
from Bio.PDB.Residue import Residue
from multiprocessing import Pool
from Bio.PDB import PDBParser
import torch
from tqdm import tqdm
warnings.filterwarnings("ignore", category=DeprecationWarning)

class GraphMaker:
    """
    Main class to generate graphs
    """

    def __init__(self, anndata_path='./adata.csv'):
        self.anndata_path = anndata_path
        self.anndata = pd.read_csv(anndata_path, sep='\t')
        self.D = {}

    def collect_data(self, x):
        """Call as a lambda function on each row of a pandas dataframe.
        Remeber to instantaite D={} as as a global var in the main script.
        
        :param x: each row of a pandas.core.DataFrame
        :type x: list
        """

        # Init lists
        dca_score = []
        pos_1 = []
        pos_2 = []

        # Grab protein names as AandB
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

            # Reset lists
            dca_score = []
            pos_1 = []
            pos_2 = []
        return

    def extract_data(self, df):
        """Modifies a globally defined dictionary with protein names,
        DCA values, and inter-protein residue positions. 

        :param df: Tao's new data
        :type df: pandas.core.DataFrame

        :return: There is no return type. Global dict (D) will be modified.
        :rtype: None
        """
        _ = df.apply(lambda x: self.collect_data(x), axis=1)
        return None

    def get_position_wise_df(self, x, protein_1, protein_2):
        """Generates an inter-protein DCA score dictionary."""

        # Intialise D dict
        _ = self.extract_data(x)
        dca_dict = {}
        protein_pair = self.D["and".join([protein_1, protein_2])]

        # Get the top 20 dca stats, if none available set to zero
        pos_1 = protein_pair['pos_1']
        pos_2 = protein_pair['pos_2']
        dca_raw = protein_pair['dca']

        # Scale DCA values
        dca = self.process_dca(dca_raw, max_value=1)
        dca_bridges = list(zip(pos_1, pos_2))
        l = list(zip(pos_1, dca))

        # Populate the dictionary
        for pos, score in l:
            if pos not in dca_dict.keys():
                dca_dict[pos] = [score]
            else:
                dca_dict[pos].append(score)

        # Take the max DCA score for a given position
        dca_dict = {}
        for k, v in dca_dict.items():
            dca_dict[k] = max(v)

        # Get residue_1 and rediue_2, and their associated DCA score cliped to DCA = 1.0
        df = pd.DataFrame({'pos_1': pos_1, 'pos_2': pos_2, 'dca': dca})
        return df, dca, dca_raw, dca_bridges

    def generate_alpha_fold_structures(self, string_to_af, pair_1, pair_2):
        """Queries alphs-fold predictions for a given protein sequence and returns
         alpha-fold predicted structure (object) for each protein in the pair.

        :param string_to_af: maping path between STRING-alphaFold
        :type string_to_af: string

        :param pair_1: protein name 1
        :type pair_1: string
        
        :param pair_2: protein name 2
        :type pair_2: string
        
        :return: alphaFold structures for each protein
        :rtype: AF-objects
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
            seq_1 and seq_2 in this case should be the same variable.
        
        # TODO: Replace remove seq_2 arg

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
        """Calculates the distance matrix for all pairwise residues,
        seq_1 and seq_2 in this case should be the same variable.

        # TODO: Replace remove seq_2 arg

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
        """Creates an adacency matrix for points within n angstroms of each other,
        seq_1 and seq_2 in this case should be the same variable.
        
        # TODO: Replace remove seq_2 arg

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

        return adjacency_matrix, contact_map

    def generate_graphs(self, adjacency_matrix_1, adjacency_matrix_2, show=False):
        """Generates the initial graphs from provided adjacency matrices,
        seq_1 and seq_2 in this case should NOT be the same variable.
        
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
                           'b-' + str(b2)
                           )
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

    def populate_edge_dca(self, graph, x, protein_1, protein_2, feature_name='dca'):
        """Populate the edge with dca score values

        :param graph: network x graph
        :type graph: nx object
        :param x: Tao's DCA data    
        :type x: pandas.core.DataFrame
        
        :param protein_1: protein 1 name    
        :type protein_1: string 
        :param protein_2: protein 2 name    
        :type protein_2: str

        :param feature_name: _description_, defaults to 'dca'
        :type feature_name: str, optional
        
        :return: nx graph with dca scores as edge feature
        :rtype: _type_
        """

        # Make data into a dataframe
        df, _, _, _ = self.get_position_wise_df(x, protein_1, protein_2)

        for edge in U.edges:
            if ('a' in edge[0]) and ('b' in edge[1]):

                # Pull out the edges with DCA connection (max DCA)
                e_1 = int(edge[0].split('-')[-1])
                e_2 = int(edge[1].split('-')[-1])

                # Extract DCA scores pertaining to the edges e1 & e2
                dca_score = df[(df['pos_1'] == e_1) & (
                    df['pos_2'] == e_2)]['dca'].values

                # Set dca edge type to dca score
                attrs = {(edge[0], edge[1]): {feature_name: float(dca_score)}}
                nx.set_edge_attributes(graph, attrs)

            else:
                # Set dca edge type to 0
                attrs = {(edge[0], edge[1]): {feature_name: 0.0}}
                nx.set_edge_attributes(graph, attrs)

        return graph

    def populate_edge_proximity(self, graph, edge_dict, feature_name='proximity'):
        """Populate none DCA edges with inverse proximity values

        :param graph: Union graph of G1 and G2
        :type graph: nx object

        :param x: edge_dict, containing {edge_id: values} as {k:v} pairs
        :type x: dictions

        :return: graph U containing edge features
        :rtype: networkx graph
        """
        
        for _, edge in enumerate(graph.edges):
            if edge[0][0] == edge[1][0]:
                attrs = {(edge[0], edge[1]): {feature_name: float(edge_dict[edge])}}
            else:
                attrs = {(edge[0], edge[1]): {feature_name: 0.0}}
            nx.set_edge_attributes(graph, attrs)
        return graph

    def process_proximity(self, x, max_value=10, mask=None, invert=True):
        """Process the proximity values via max value division and invertion (1-value)

        :param x: proximity scores
        :type x: list of ints

        :param max_value: the maximum value in the lest, defaults to 10
        :type max_value: int, optional
        
        :param mask: to mask out any edges that connect residues > 10 angstrom, defaults to None
        :type mask: _type_, optional
        
        :param invert: if true, a score of 0.x becomes 1-0.x, defaults to True
        :type invert: bool, optional

        :return: generates processed proximity scores   
        :rtype: np.array
        """

        # Scale proximity values to max_value
        x = copy.deepcopy(x)
        x_array = np.array(x) / max_value

        # To mask background edges (geater than 10 angstroms)
        if mask is None:
            mask = np.ones(np.shape(x))

        # Apply mask
        x_array = mask * x_array

        # Invert: small scores should become large
        if invert:
            x_array = 1 - x_array
        return x_array

    def get_proximity_dict(self, x, graph_name='a'):
        """Converts proximity scores into a dictionary"""
        row, col = x.shape
        edge_d = {}
        for r in range(row):
            for c in range(col):
                key = ('{}-{}'.format(graph_name, r)
                       ), ('{}-{}'.format(graph_name, c))
                if key not in edge_d.keys():
                    val = x[r, c]
                    edge_d[key] = val
        return edge_d

    def process_dca(self, x, max_value=1):
        """Scales DCA values

        :param x: DCA scores
        :type x: list

        :param max_value: maximum value to clip DCA scores to, defaults to 1
        :type max_value: int, optional
        
        :return: clipped values
        :rtype: np.array
        """
        # Scale DCA values
        x = copy.deepcopy(x)
        x_array = np.array(x)
        x_array[x_array > 1] = max_value
        return x_array

    def save_graph_data(self, graph, protein_1, protein_2, label):

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

        nx.write_gpickle(graph, os.path.join(
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
    
    def get_computational_graph(self, string_to_af, coevo_path, netsurf_d, n_samples=-1):
        """Higher level class method to generate all graphs in the dataset.

        :param string_to_af: maping path between STRING-alphaFold
        :type string_to_af: string
        :param coevo_path: path to coevolutionary data containing DCA scores    
        :type coevo_path: string
        :param netsurf_d: dictionary contianing netsurf features keyed by protein names 
        :type netsurf_d: dictionary
        :param n_samples: number of samples to generate from anndata object (-1 = all available), defaults to -1
        :type n_samples: int, optional
        """
        
        
        observations = self.anndata.iloc[:n_samples, :]
        n_rows, _ = np.shape(observations)

        # Loop over all instances in the anndata (annotation data) file
        for i in tqdm(range(n_rows)):

            # NOTE - try/except: If there is an issue at any point for a given graph generation...
            #  completely exclude it from the caching process.Usually 4 errors per hundred iterations.

            # TODO - Catch these exceptions smarter and examine issue - sometimes proteins listed in the Anndata
            #        ... object do not contain a coevolution/DCA score. 

            try:
                obs = observations.iloc[i, :]
                pair_1 = obs['STRING_ID1']
                pair_2 = obs['STRING_ID2']
                label = obs['benchmark_status']

                # Get alpha-fold structures
                residue_1, residue_2 = self.generate_alpha_fold_structures(
                    string_to_af, pair_1, pair_2)

                # Get proximity matrices
                proximity_mask_1, dist_map_1 = self.generate_proximity_matrix(
                    seq_1=residue_1, seq_2=residue_1,
                    angstroms=10, show=False)

                proximity_mask_2, dist_map_2 = self.generate_proximity_matrix(
                    seq_1=residue_2, seq_2=residue_2,
                    angstroms=10, show=False)

                # Generate graphs
                G_1, G_2 = self.generate_graphs(
                    adjacency_matrix_1=proximity_mask_1,
                    adjacency_matrix_2=proximity_mask_2)

                # Populate node attributes with netsurfp features
                G_1, G_2 = self.populate_graph_features(
                    graph_1=G_1, graph_2=G_2,
                    protein_1=pair_1, protein_2=pair_2,
                    netsurf_path_dict=netsurf_d)

                # Get DCA bridge connections
                df, dca, dca_raw, dca_bridges = self.get_position_wise_df(
                    x=coevo_path, protein_1=pair_1,
                    protein_2=pair_2)

                # Make union of the graphs on dca brides
                U = self.link_graphs(
                    graph_1=G_1, graph_2=G_2,
                    dca_bridges=dca_bridges, show=False)

                # Populate edge attributes
                U = self.populate_edge_dca(
                    graph=U, x=coevo_path,
                    protein_1=pair_1, protein_2=pair_2,
                    feature_name='dca')

                # Invert distance matrices
                inv_dit_map_1 = self.process_proximity(
                    x=dist_map_1, max_value=10,
                    mask=proximity_mask_1, invert=True)

                inv_dit_map_2 = self.process_proximity(
                    x=dist_map_2, max_value=10,
                    mask=proximity_mask_2, invert=True)

                # Generate proximity matrices
                d_1 = self.get_proximity_dict(x=inv_dit_map_1, graph_name='a')
                d_2 = self.get_proximity_dict(x=inv_dit_map_2, graph_name='b')

                # Combine the two dicts
                d_1.update(d_2)

                # Finally populate the prixmity edge features
                U = self.populate_edge_proximity(U, d_1)

                # Save the graph to file
                self.save_graph_data(graph=U, protein_1=pair_1,
                                protein_2=pair_2, label=label)
            except:
                print('Skipping... something went wrong - continuing from next instance.')
                pass
            
class SloppyStructureBuilder(Bio.PDB.StructureBuilder.StructureBuilder):
    """Cope with resSeq < 10,000 limitation by just incrementing internally.

    # Q: What's wrong here??
    #   Some atoms or residues will be missing in the data structure.
    #   WARNING: Residue (' ', 8954, ' ') redefined at line 74803.
    #   PDBConstructionException: Blank altlocs in duplicate residue SOL
    #   (' ', 8954, ' ') at line 74803.
    #
    # A: resSeq only goes to 9999 --> goes back to 0 (PDB format is not really
    #    good here)
    """

    # NOTE/TODO:
    # - H and W records are probably not handled yet (don't have examples
    #   to test)

    def __init__(self, verbose=False):
        Bio.PDB.StructureBuilder.StructureBuilder.__init__(self)
        self.max_resseq = -1
        self.verbose = verbose

    def init_residue(self, resname, field, resseq, icode):
        """Initiate a new Residue object.

        Arguments:
        o resname - string, e.g. "ASN"
        o field - hetero flag, "W" for waters, "H" for
            hetero residues, otherwise blanc.
        o resseq - int, sequence identifier
        o icode - string, insertion code

        """
        if field != " ":
            if field == "H":
                # The hetero field consists of
                # H_ + the residue name (e.g. H_FUC)
                field = "H_" + resname
        res_id = (field, resseq, icode)

        if resseq > self.max_resseq:
            self.max_resseq = resseq

        if field == " ":
            fudged_resseq = False
            while self.chain.has_id(res_id) or resseq == 0:
                # There already is a residue with the id (field, resseq, icode)
                # resseq == 0 catches already wrapped residue numbers which
                # do not trigger the has_id() test.
                #
                # Be sloppy and just increment...
                # (This code will not leave gaps in resids... I think)
                #
                # XXX: shouldn't we also do this for hetero atoms and water??
                self.max_resseq += 1
                resseq = self.max_resseq
                res_id = (field, resseq, icode)  # use max_resseq!
                fudged_resseq = True

            if fudged_resseq and self.verbose:
                sys.stderr.write(
                    "Residues are wrapping (Residue "
                    + "('%s', %i, '%s') at line %i)."
                    % (field, resseq, icode, self.line_counter)
                    + ".... assigning new resid %d.\n" % self.max_resseq
                )
        residue = Residue(res_id, resname, self.segid)
        self.chain.add(residue)
        self.residue = residue

class SloppyPDBIO(Bio.PDB.PDBIO):
    """PDBIO class that can deal with large pdb files as used in MD simulations

    - resSeq simply wrap and are printed modulo 10,000.
    - atom numbers wrap at 99,999 and are printed modulo 100,000

    """

    # The format string is derived from the PDB format as used in PDBIO.py
    # (has to be copied to the class because of the package layout it is not
    # externally accessible)
    _ATOM_FORMAT_STRING = (
        "%s%5i %-4s%c%3s %c%4i%c   " + "%8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n"
    )

    def _get_atom_line(
        self,
        atom,
        hetfield,
        segid,
        atom_number,
        resname,
        resseq,
        icode,
        chain_id,
        element="  ",
        charge="  ",
    ):
        """Returns an ATOM string that is guaranteed to fit the ATOM format.

        - Resid (resseq) is wrapped (modulo 10,000) to fit into %4i (4I) format
        - Atom number (atom_number) is wrapped (modulo 100,000) to fit into
          %5i (5I) format

        """
        if hetfield != " ":
            record_type = "HETATM"
        else:
            record_type = "ATOM  "
        name = atom.get_fullname()
        altloc = atom.get_altloc()
        x, y, z = atom.get_coord()
        bfactor = atom.get_bfactor()
        occupancy = atom.get_occupancy()
        args = (
            record_type,
            atom_number % 100000,
            name,
            altloc,
            resname,
            chain_id,
            resseq % 10000,
            icode,
            x,
            y,
            z,
            occupancy,
            bfactor,
            segid,
            element,
            charge,
        )
        return self._ATOM_FORMAT_STRING % args

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super(GCN, self).__init__()

        # Parameters
        self.num_node_features = 16
        self.num_classes = 1
        self.hidden_channels = hidden_channels

        # Layers (consider using class SAGEConv instead)
        self.conv1 = GCNConv(self.num_node_features, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.linear_1 = Linear(self.hidden_channels, self.hidden_channels)
        self.linear_2 = Linear(self.hidden_channels, self.num_classes)

        # Paramteric RelU
        self.prelu_1 = torch.nn.PReLU()
        self.prelu_2 = torch.nn.PReLU()
        self.prelu_3 = torch.nn.PReLU()
        self.prelu_4 = torch.nn.PReLU()

        # Regularization
        self.batch_norm_1 = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels, track_running_stats=False, momentum=None)

        self.batch_norm_2 = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels, track_running_stats=False, momentum=None)

        self.batch_norm_3 = torch.nn.BatchNorm1d(
            num_features=self.hidden_channels, track_running_stats=False, momentum=None)

        self.batch_norm_4 = torch.nn.BatchNorm1d(
            num_features=self.num_classes, track_running_stats=False, momentum=None)

    def forward(self, x, edge_index, batch):

        # 1.Conv block 1
        x = self.conv1(x, edge_index)
        x = self.batch_norm_1(x)
        x = self.prelu_1(x)

        # Conv block 2
        x = self.conv2(x, edge_index)
        x = self.batch_norm_2(x)
        x = self.prelu_2(x)

        # Pool data across the rows
        x = global_max_pool(x, batch)  # --> [batch_size, hidden_channels]

        # 3. Linearization
        x = self.linear_1(x)
        x = self.batch_norm_3(x)
        x = self.prelu_3(x)

        # 4. Logic outputs
        x = self.linear_2(x)
        x = self.batch_norm_4(x)
        x = self.prelu_4(x)
        return x






