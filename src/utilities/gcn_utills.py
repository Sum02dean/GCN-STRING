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
import glob
import shutil
import Bio.PDB
import Bio.PDB.StructureBuilder
from Bio.PDB.Residue import Residue
from multiprocessing import Pool
from Bio.PDB import PDBParser
import torch
from tqdm import tqdm
from numpy.linalg import matrix_rank
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FrequencyCoupler:
    """Class for computing and visualizing the freuqency matrix of top-n DCA
       aligned inter-protein residue pairs. This class is used to compute DCA scores on combined protein MSA .fasta files. 
       The frequency matix can also be computed alonside a visualization.
    """

    def __init__(self, msa_file_path, anndata_path, cache_dir_path):
        self.cache_dir_path = cache_dir_path
        self.msa_file_path = msa_file_path
        self.anndata = pd.read_csv(anndata_path, sep='\t')
        self.tokens = ['C', 'M', 'W',
                       'F', 'Y', 'H',
                       'P', 'V', 'S',
                       'T', 'D', 'E',
                       'N', 'Q', 'I',
                       'L', 'R', 'K',
                       'A', 'G', 'X', '-']
        # Sorted based on QSAR properties

    def get_pair_id(self, fn):
        fn = fn.replace('.fasta', '')
        sfn = fn.split('/')[-1]
        sfn_m = sfn.split('and')
        return sfn_m

    def get_dca_scores(self, pseudocount=0.5, seqid=0.8, plen1=None):

        # Get fasta name
        fasta_name = self.msa_file_path.split('/')[-1]
        pair_id = self.get_pair_id(self.msa_file_path)

        if plen1:
            self.plen1 = plen1
        else:
            plen1 = int(self.anndata.loc[(self.anndata['STRING_ID1'] == pair_id[0])
                                         & (self.anndata['STRING_ID2'] == pair_id[1])]['len1'])
            plen2 = int(self.anndata.loc[(self.anndata['STRING_ID1'] == pair_id[0])
                                         & (self.anndata['STRING_ID2'] == pair_id[1])]['len2'])
            self.plen1 = plen1
            self.plen2 = plen2

        # Explore all files in the cache directory
        files = glob.glob(os.path.join(self.cache_dir_path, '*'))
        f = os.path.join(self.cache_dir_path,
                         fasta_name.replace('.fasta', '.npy'))

        # Check to see if the speicific dca file is cached
        file_identified = False
        for fn in files:
            if fasta_name.replace('.fasta', '.npy') in fn:
                file_identified = True
                self.mfdca_FN_APC = np.load(f, allow_pickle=True)

        if file_identified == False:
            # If file not in cache, compute as usual using pyDCA
            self.mfdca_inst = meanfield_dca.MeanFieldDCA(self.msa_file_path,
                                                         'protein',
                                                         pseudocount=pseudocount,
                                                         seqid=seqid)

            # Compute average product corrected Frobenius norm of the couplings
            self.mfdca_FN_APC = self.mfdca_inst.compute_sorted_FN_APC()

            # Save results to cache
            with open(f, 'wb') as fn:
                np.save(f, self.mfdca_FN_APC)

        # Get *inter* pairs
        p1 = []
        p2 = []
        s = []
        for pairs, scores in self.mfdca_FN_APC:
            # Subtract 1 from plen1 to account for 0th indexing
            if (pairs[0] <= plen1 - 1) and (pairs[1] > plen1 - 1):
                p1.append(pairs[0])
                p2.append(pairs[1])
                s.append(scores)

        # Store *inter*
        inter_protein_pairs = np.array(list(zip(p1, p2, s)))

        # Get *intra* pairs for protein A
        p1 = []
        p2 = []
        s = []
        for pairs, scores in self.mfdca_FN_APC:
            # Subtract 1 from plen1 to account for 0th indexing
            if (pairs[0] <= plen1 - 1) and (pairs[1] <= plen1 - 1):
                p1.append(pairs[0])
                p2.append(pairs[1])
                s.append(scores)

        # Store intra** A
        intra_protein_pairs_a = np.array(list(zip(p1, p2, s)))

        # Get intra** pairs for protein B
        p1 = []
        p2 = []
        s = []
        for pairs, scores in self.mfdca_FN_APC:
            # Subtract 1 from plen1 to account for 0th indexing
            if (pairs[0] > plen1 - 1) and (pairs[1] > plen1 - 1):
                p1.append(pairs[0])
                p2.append(pairs[1])
                s.append(scores)

        # Store intra** B
        intra_protein_pairs_b = np.array(list(zip(p1, p2, s)))

        # Return arrays
        return inter_protein_pairs, intra_protein_pairs_a, intra_protein_pairs_b

    def convert_msa_to_df(self):
        """Converts a paired_msa file into a pandas DataFrame where columns represent residue postion
           each row is the protein sequence of a multiple-sequence-alignmnt fasta file.

        :return: pandas DataFrame object
        :rtype: object
        """
        # Convert protein sequence into an amino acid dataframe for protein
        bio_rec = SeqIO.parse(self.msa_file_path, "fasta")
        seq = []
        for _, rec in enumerate(bio_rec):
            sequence = rec.seq
            seq.append([x for x in sequence])
        df = pd.DataFrame(seq)
        return df

    def get_consensus_matrix(self, threshold):
        """Computes the consensus sequence for a multi-squence alignment.

        :param threshold: the threshold value that is required to add a particular atom.
        :type threshold: float/int
        :return: Returns a fast consensus sequence of the alignment
        :rtype: object
        """

        align = AlignIO.read(
            self.msa_file_path, "fasta")
        summary_align = AlignInfo.SummaryInfo(align)
        consensus = summary_align.dumb_consensus(threshold)
        self.consensus = consensus
        return consensus

    def subset_top_dca_couplings(self, top_inter_pair):
        """Subsets the MSA to return a minimal pandas DataFrame containing only the aligned inter-protein residue with the top DCA score.

        :param top_inter_pair: the residue pair (two pandas columns) with the highest DCA score(s).
        :type top_inter_pair: double
        :return: a pandas DataFrame with two columns corresponding to the residue pair with the highest score across the MSA.
        :rtype: pandas DataFrame
        """
        msa_df = self.convert_msa_to_df()
        top_row_a = msa_df.iloc[:, int(top_inter_pair[0])]
        top_row_b = msa_df.iloc[:, int(top_inter_pair[1])]
        top_pairs = pd.concat([top_row_a, top_row_b], axis=1)
        self.top_pairs = top_pairs
        return top_pairs

    def get_frequency_matrix(self, top_msa_pair, plot=False):
        """Generates the amino acid frequency matrix for the top DCA residues positions across MSA

        :param top_msa_pair: the actual MSA residues from top-scoring pair column ids (two pandas columns).
        :type top_inter_pair: pandas df
        :return: inter-protein frequeny matrix for highest DCA scoring pair
        :rtype: numpy array

        """

        # This is essentially the empty frequency matrix
        couplings = {}
        # For token A and B in pairwise map
        for token_a in self.tokens:
            for token_b in self.tokens:
                keys = token_a + token_b
                coupling_count = 0

                # Record the coupling freuqency for each instance that occurs in the pairwise map
                for pair in top_msa_pair.values:
                    # Increment counter if the coulings are contained in the pairwise map
                    if (pair[0] == keys[0]) and (pair[1] == keys[1]):
                        coupling_count += 1
                    # Generate a couplings report
                    couplings.update({keys: coupling_count})

        # Reshape and plot:
        vals = np.array([x for x in couplings.values()])
        coupling_matrix = np.reshape(
            vals, (len(self.tokens), len(self.tokens)))

        # Normalize the coupling matrix by MSA-depth
        msa_depth = np.shape(top_msa_pair)[0]
        coupling_matrix = coupling_matrix / msa_depth
        self.coupling_matrix = coupling_matrix

        # To visuliase or not
        if plot:
            self.visualize_frequency_matrix()

        return coupling_matrix

    def visualize_frequency_matrix(self):
        """Simply visualizes the frequency matrix for the top DCA scoring inter-protein residue positions.
        """

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(self.coupling_matrix, mask=None, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        # Plot descriptions
        ax.set_xticklabels(self.tokens)
        ax.set_yticklabels(self.tokens)
        plt.title("Resiude-wise Frequency Matrix for Top Scoring DCA Coupling")
        plt.ylabel("Top Residue Position in Protein A")
        plt.xlabel("Top Residue Position in Protein B")
        return f

    def get_file_name(self):
        file_name = self.msa_file_path.split('/')[-1]
        f = file_name.replace(".fasta", "")
        return f


class DcaLab:
    def __init__(self, anndata_path):
        """Contains the tools for calculating DCA workflows.

        :param anndata_path: path to "all_info..." file.
        :type anndata_path: string
        """
        # Generate annotation datafiles
        self.anndata_path = anndata_path
        self.anndata = pd.read_csv(anndata_path, sep='\t')

    def make_triangular_dca_array(self, file_name):
        """Converts raw DCA *.npy files into an upper triangle DCA contact map

        :param file_name: DCA numpy file.
        :type anndata_path: string

        :return: The upper triangle DCA contact map
        :rtype: numpy array
        """

        # Load in the dca score for a paired MSA
        x = np.load(file_name, allow_pickle=True)
        ident = file_name.replace('.npy', '')
        ident = ident.split('/')[-1]
        pair_id = self.get_pair_id(fn=file_name)

        # Extract the protein lengths
        plen_1 = int(self.get_anndata_field(field='len1', pair_id=pair_id))
        plen_2 = int(self.get_anndata_field(field='len2', pair_id=pair_id))

        # Re-format the dca scores into a triangular matrix
        dca_array = np.zeros((plen_1 + plen_2, plen_1 + plen_2))
        row_idx = np.array([int(t[0]) for t in x[:, 0]])
        col_idx = np.array([t[1] for t in x[:, 0]])

        # Populate the upper triangular matrix
        dca_scores = np.array([t for t in x[:, 1]])
        dca_array[row_idx, col_idx] = dca_scores
        return dca_array

    def get_string_2_pdp_list(self, dist_path):
        """ Get the mapping of STRING-PDB files

        :param dist_path: _description_
        :type dist_path: _type_
        :return: _description_
        :rtype: _type_
        """
        # Import all String-PDB mapped files
        with open(dist_path, 'rb') as handle:
            str_2_pdb = pickle.load(handle)
        return str_2_pdb

    def get_pair_id(self, fn):
        """Extracts the name of protein-A and protein-B pairs from the file_name

        :param fn: The filename which contains the protein identities
        :type fn: string
        :return: The two protein identifiers from the file_name string
        :rtype: list
        """
        # Extrac the combined protein ID
        fn = fn.replace('.npy', '')
        sfn = fn.split('/')[-1]
        sfn_m = sfn.split('and')
        return sfn_m

    def get_anndata_field(self, field='', pair_id=[0, 0]):
        """Extracts a column from a CSV file based on column name at a specific row

        :param field: column name, defaults to 'STRING_ID1'
        :type field: str, optional
        :param pair_id: the two protein identifiers for the specific file, defaults to [0, 0]
        :type pair_id: list, optional
        :return: the subset data belonging to the column name given by "field"
        :rtype: Series
        """
        # Subset the field from the row specified by pair_id
        if field == '':
            print("Please provide a valid value")
            f = None
        else:
            f = self.anndata.loc[(self.anndata['STRING_ID1'] == pair_id[0])
                                 & (self.anndata['STRING_ID2'] == pair_id[1])][field]
        return f

    def create_dictionary(self, dist_path, file_path, get_dca=True, suffix=".npy"):
        """Returns two dictionary objects, keys are the STRING-PDB mapped pairs, keys are
           the symmetrized contact scores, either DCA or known contacts (see get_dca param).

        :param distance_path: This is the path location for known distance values
        :type distance_path: string
        :param file_path: This is the path to the DCA files
        :type file_path: string
        :param get_dca: if file_path pertains to  DCA files, defaults to True
        :type get_dca: bool, optional
        :param suffix: The file suffix to add onto the end of the par_id, defaults to ".npy"
        :type suffix: str, optional
        :return: the inter and intra protein dictionaries contaning symmetrical contact matrices
        :rtype: [dict, dict]
        """

        # Import all String-PDB mapped files
        str_2_pdb = self.get_string_2_pdp_list(dist_path)
        # Instantiate empty dictionaries for inter and intra DCA matrices
        intra_dict = dict()
        inter_dict = dict()

        for _, (str_id_1, str_id_2, pdb_id_1, pdb_id_2, _) in enumerate(str_2_pdb):

            # This try clause will load in all available benchmarsk - it is a speed bottleneck!
            try:
                # Get the STRING-ID from the mapped str_2_pdb list
                string_id = "and".join((str_id_1, str_id_2))
                # Get the PDB -D from the mapped str_2_pdb list
                pdb_id = "and".join((pdb_id_1, pdb_id_2))

                if get_dca:
                    # Get the name of the dca file and import - this should really be upper triangle
                    filename = os.path.join(file_path, string_id + suffix)
                    array = np.load(filename, allow_pickle=True)

                    if array.shape[0] != array.shape[1]:
                        # If not triangular, convert dca_scores to upper triangle
                        array = self.make_triangular_dca_array(filename)

                else:
                    # Get known physical distace file - should be upper triangle
                    filename = os.path.join(file_path, pdb_id + suffix)
                    array = np.load(filename, allow_pickle=True)

                # Grab the length of all eligible STRING protein pairs
                pairs = (str_id_1, str_id_2)

                # Protein  lengths
                L1 = int(self.get_anndata_field(
                    pair_id=pairs, field='len1'))
                L2 = int(self.get_anndata_field(
                    pair_id=pairs, field='len2'))

                # Create the symetrical matrix from the imported file
                symmetric_array = array.T + array
                np.fill_diagonal(symmetric_array, 0)

                # Populate the dictionaries
                intra_dict[pairs] = symmetric_array
                inter_dict[pairs] = symmetric_array[: L1, L1:]

            except Exception as e:
                print(e)
                # There are two reasons why a loop my pass, error, or lack of String-PDB mapping.
                pass

            # Make available to all methods
            self.intra_dict = intra_dict
            self.inter_dict = inter_dict

        return intra_dict, inter_dict

    def get_local_matrix(self, pairs, path, suffix=".npy"):
        """Creates two local matrices, one containing the inter-protein contacts, 
            the other containing the intra-protein contacts. 

        :param pairs: String ID for the twp interacting proteins
        :type pairs: list of strings
        :param path: the path to the directory containing DCA files
        :type path: str
        :param suffix: the end of the file name (after the string or PDB ID's), defaults to ".npy"
        :type suffix: str, optional
        """

        # Extract protein pair IDs and protein lengths
        id_1, id_2 = pairs
        L1 = int(self.get_anndata_field(pair_id=pairs, field='len1'))

        # Import matrix file
        file_name = os.path.join(path, id_1 + "and" + id_2 + suffix)
        data_array_dig = np.load(file_name, allow_pickle=True)

        # If not triangular, convert matrix to upper triangle
        if data_array_dig.shape[0] != data_array_dig.shape[1]:
            data_array_dig = self.make_triangular_dca_array(file_name)

        # Symmetrize the matrix
        data_array = data_array_dig.T + data_array_dig
        np.fill_diagonal(data_array, 0)

        # Extract the relevent localities: Inter-protein contacts
        inter_array = copy.deepcopy(data_array_dig[: L1, L1:])
        maxRow, maxCol = np.unravel_index(
            inter_array.argmax(), inter_array.shape)
        maxCol += L1

        # Extract the relevent localities: Intra-protein contacts
        intra_array = np.zeros((data_array.shape))
        intra_array[0:L1, 0:L1] = data_array[0:L1, 0:L1]
        intra_array[L1:, L1:] = data_array[L1:, L1:]

        data_array[::] = 0
        data_array[:L1, L1:] = inter_array
        return(maxRow, maxCol, data_array, intra_array)

    def plot_scatter_heatmap(self, pairs, dca_path=None, dist_path=None):
        """Plots the contact map between two protein pairs based on DCA computations.
            Top-100 scoring proteins are shown in blue, top-20 in red.

        :param pairs: protein pairs
        :type pairs: list of str
        :param dca_path: path to directory containing DCA files, defaults to None
        :type dca_path: str, optional
        :param dist_path: path to file containing physical distance files, defaults to None
        :type dist_path: str, optional
        """
        # Extract protein pair IDs
        L1 = int(self.get_anndata_field(pair_id=pairs, field='len1'))
        L2 = int(self.get_anndata_field(pair_id=pairs, field='len2'))

        # Get DCA localities
        maxRow, maxCol, inter_array, intra_array = self.get_local_matrix(
            pairs=pairs, path=dca_path)

        # Flatten inter-DCA arrays
        inter_array_flat = inter_array.flatten()
        inter_array_flat.sort()

        # Flatten intra-DCA arrays
        intra_array_flat = intra_array.flatten()
        intra_array_flat.sort()

        # Get data-points with top-100 DCA scores
        idx_rows, idx_cols = np.where(intra_array >= intra_array_flat[-100])

        # Create axis lines
        fig = plt.figure()
        plt.xlim(0, inter_array.shape[0])
        plt.ylim(0, inter_array.shape[1])
        plt.axhline(y=L1, linewidth=0.3, color='g', linestyle="-.")
        plt.axvline(x=L1, linewidth=0.3, color='g', linestyle="-.")

        # Modify tick fonts
        plt.xticks(np.arange(0, L1 + L2, step=20), fontsize=1)
        plt.yticks(np.arange(0, L1 + L2, step=20), fontsize=1)

        # Visualise top-100 DCA points
        plt.plot(idx_rows, idx_cols, color="g", marker=".",
                 linestyle='None', markersize=1)

        # Modify ticks, top 100 are blue, top 20 are red
        for tick, c in [(-100, 'b'), (-20, 'r')]:
            idx_rows, idx_cols = np.where(
                inter_array >= inter_array_flat[tick])
            plt.plot(idx_rows, idx_cols, color=c, marker=".",
                     linestyle='None', markersize=1)
            plt.text(maxRow, maxCol, "max", fontsize=5)
            plt.xlim(0, inter_array.shape[0])
            plt.ylim(0, inter_array.shape[1])

        # Plot phycial distance
        phys_pairs = pairs
        dist_thres = 30

        # Generate intra-protein dict if it doesn't exist
        self.intra_dict, _ = self.create_dictionary(
            dist_path=dist_path, file_path=dca_path)
        return

    def generate_frequency_coupler(self, msa_file_path, cache_dir_path='/configs'):
        """Retrieves a frequency_coupler instance

        :param msa_file_path: path to the paired_msa alignment file
        :type msa_file_path: string
        :return: an instance of FrequencyCoupler()
        :rtype: object
        """
        fc = FrequencyCoupler(
            anndata_path=self.anndata_path, msa_file_path=msa_file_path, cache_dir_path=cache_dir_path)
        return fc

    def generate_msa_synthesizer(self):
        """Returns an instance of MsaSynthesizer.

        :return: MsaSynthesizer instance
        :rtype: object
        """
        msa_synthesizer = MsaSynthesizer()
        return msa_synthesizer


class FileParser:
    """File parser to grab co-evolution, entropy and annotation data for protein pairs.
        This is a class designed to parse the blastp file directory found in Taos remote
        location "/mnt/mnemo6/tao/PPI_Coevolution/ecoli_size2complex_new/blastp".
        The aim is to take any file matching a specific pattern and to collate it into a list.
        These files can then be moved or copied en bulk to a new location.
    """

    def __init__(self, root_path="/mnt/mnemo1/sum02dean/dean_mnt/projects/blastp"):
        """
        :param root_path: path to root directory "/mnt/mnemo6/tao/PPI_Coevolution/ecoli_size2complex_new/blastp/"
        :type root_path: str, optional
        """
        self.root_path = root_path

    def get_root_path(self):
        """Basic getter
        :return: root path
        :rtype: string
        """
        return self.root_path

    def get_data(self, category='coevolution'):
        """Grabs the data of protein pairs according to the given category.
        :param category: one of 'coevolution', 'entropy' or 'annotation', defaults to 'coevolution'
        :type category: str, optional
        :return: a list of all of the file names for a given category
        :rtype: list
        """
        if category == 'coevolution':
            pattern = '*_pydcaFNAPC_array.npy'

        elif category == 'entropy':
            pattern = '*_apc_allResidues.npy'

        file_list = []
        dir_name = os.path.join(self.root_path, pattern)
        files = glob.glob(dir_name)
        for name in files:
            if name.find(pattern):
                file_list.append(name)
        return file_list

    def copy_data(self, file_list, dest_dir):
        """Method for copying data from one location to another.
            :param file_list: list of files to be copied using relative path, to generate this list see .get_data() method.
            :type file_list: list
            :param dest_dir: copy destination path for all provided files in file_list.
            :type dest_dir: string
            """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for f in file_list:
            shutil.copy(f, dest_dir)

    def move_data(self, file_list, dest_dir):
        """Method for moving data from one location to another.
            :param file_list: list of files to be moved using relative path, to generate this list see .get_data() method.
            :type file_list: list
            :param dest_dir: move destination path for all provided files in file_list.
            :type dest_dir: string
            """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for f in file_list:
            shutil.move(f, dest_dir)


class GraphMaker:
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
        data_file_name = coevo_path + query_pro1 + "and" + query_pro2 + suffix + ".npz"
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
                    if len(return_list) < (2 + top_num * 3):
                        return_list.extend((top_value, top_row, top_col))
                        scores.append(top_value)
                        pairs_1.append(top_row)
                        pairs_2.append(top_col)

                esisted_top_values.add(top_value)
            if len(return_list) > (2 + top_num * 3):
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
            structure_builder=SloppyStructureBuilder())
        protein_1 = sloppy_parser.get_structure(id=None, file=test_file_1)
        protein_2 = sloppy_parser.get_structure(id=None, file=test_file_2)

        # Structure 1
        sloppyio_1 = SloppyPDBIO()
        sloppyio_1.set_structure(protein_1)

        # Structure 2
        sloppyio_2 = SloppyPDBIO()
        sloppyio_2.set_structure(protein_2)

        # (Optional) save the sloppyIO files
        # sloppyio_1.save("sloppyio_1.pdb")
        # sloppyio_2.save("sloppyio_2.pdb")

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
        seq_pos_p2 = [x + (seq_pos_p1.values[-1] + 1)
                      for x in net_surf_2.seqPos]
        msa_pos_p2 = [x + (msa_pos_p1.values[-1] + 1)
                      for x in net_surf_2.MSAPos]
        msa_2_seq_p2 = dict(zip(msa_pos_p2, seq_pos_p2))
        seq_2_msa_p2 = dict(zip(seq_pos_p2, msa_pos_p2))

        # Reset G2 labels
        g2_labels = [x + (graph_1_labels[-1]) for x in x_net_surf_2.n.values]
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
                len_1 = int(self.anndata.loc[(self.anndata['STRING_ID1'] == pair_1) & (
                    self.anndata['STRING_ID2'] == pair_2)].len1)
                len_2 = int(self.anndata.loc[(self.anndata['STRING_ID1'] == pair_1) & (
                    self.anndata['STRING_ID2'] == pair_2)].len2)
                label = self.anndata.loc[(self.anndata['STRING_ID1'] == pair_1) & (
                    self.anndata['STRING_ID2'] == pair_2)].benchmark_status.values
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

                # Create graph_data dir if not present
                graph_folder_name = 'graph_data'
                isExist = os.path.exists(graph_folder_name)
                if not isExist:
                    # Create it
                    os.makedirs(graph_folder_name)
                    print("{} directory created.".format(graph_folder_name))

                labels_folder_name = 'graph_labels'
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


class MsaSynthesizer:
    """
        This class is responsible for artificially generating an MSA
        with some co-evolutionary contraints - used to test the pyDCA implimentation.
    """

    def __init__(self):
        self.tokens = [
            '-', 'A', 'C',
            'D', 'E', 'F',
            'G', 'H', 'I',
            'K', 'L', 'M',
            'N', 'P', 'Q',
            'R', 'S', 'T',
            'V', 'W', 'X', 'Y']

        # Instantiate the ASCII to char coders
        self.encoder = self.__build_encoder()
        self.decoder = self.__build_decoder()
    # Class methods

    def generate_random_msa(self, len_msa=100, dep_msa=10, mutate_threshold=0.8):
        # Generate ASCII MSA and select random column pair
        ascii_msa = self.__generate_ascii_msa(len_msa, dep_msa)
        pair = self.__select_random_pairs(n=2, len_msa=len_msa)

        # Mutate the random columns in a synchronous manner
        covariate_ascii_msa = self.__generate_covariate_pair(x=ascii_msa,
                                                             pair=pair, mutate_threshold=mutate_threshold)

        # Convert ASCII MSA to character MSA
        msa = self.__decode_ascii_matrix_to_char_matrix(covariate_ascii_msa)
        return pair, msa

    def convert_msa_to_alignment(self, df):
        """ Create a fasta MSA alignment using biopython

        :param df: a dataframe containing amino acid character encodings
        :type df: pandas DataFrame
        :return: seqAlign object
        :rtype: obect
        """
        seqs = ["".join(x) for x in df.values.tolist()]
        alignments = MultipleSeqAlignment(
            [], Gapped(SingleLetterAlphabet, "-"))

        for i, seq in enumerate(seqs):
            alignments.add_sequence(str(i), seq)
        return alignments

    def save_to_fasta(self, alignments, file_name):
        """Writes a Bio segAlign MSA object to fasta file.

        :param alignments: alignment objects
        :type alignments: Bio object
        :param file_name: path/filename - to save fasta file
        :type file_name: string
        """
        with open(file_name, "w") as handle:
            SeqIO.write(alignments, handle, "fasta")
            handle.close()

    def __char_to_ascii(self, x):
        """ Decodes character to ascii float

        :param x: character to decoder
        :type x: string
        :return: the ascii coe mapped to th string character
        :rtype: float
        """
        return self.encoder[x]

    def __decode_ascii_matrix_to_char_matrix(self, x):
        """ Takes an ascii matrix and decodes it to a character matrix of amino acids.

        :param x: ascii n-d array
        :type x: numpy array
        :return: pandas DataFrame of amino acid tokens
        :rtype: pandas DataFrame
        """
        df = pd.DataFrame(x)
        df = df.applymap(self.__ascii_to_char)
        return df

    def __encode_char_to_ascii_matrix(self, df):
        """ Takes an char matrix and encodes it to an ascii matrix.

        :param x: ascii n-d array
        :type x: numpy array
        :return: pandas DataFrame of amino acid tokens
        :rtype: pandas DataFrame
        """
        df = df.applymap(self.__char_to_ascii)
        return df

    def __ascii_to_char(self, x):
        """ Encodes ascii characters to character string

        :param x: ascii character
        :type x: float
        :return: the character mapped to ascii code
        :rtype: string
        """
        return self.decoder[x]

    def __generate_ascii_msa(self, len_msa, dep_msa):
        """ Generates a random msa matrix in ascii format.

        :param len_msa: the number of resiues in the msa matrix
        :type len_msa: int
        :param dep_msa: the depth of the msa (amount of sequences)
        :type dep_msa: int
        :return: a random protein msa
        :rtype: numpy array
        """
        msa = np.empty((dep_msa, len_msa))
        for i in range(0, dep_msa):
            for j in range(0, len_msa):
                msa[i, j] = self.__char_to_ascii(np.random.choice(self.tokens))
        return msa

    def __select_random_pairs(self, n=2, len_msa=10):
        """ Simply selects n random columns in the msa for a given length of

        :param n: number of columns to return, defaults to 2
        :type n: int, optional
        :param len_msa: the length of the msa, defaults to 10
        :type len_msa: int, optional
        :return: the indexes of the two chosen columns
        :rtype: list
        """
        pairs = random.sample(range(0, len_msa), n)
        sorted_pairs = np.sort(pairs)
        return [x for x in sorted_pairs]

    def __generate_covariate_pair(self, x, pair=[0, 0], mutate_threshold=0.8):
        """Generate a pairwise co-evolving "point of contact"

        :param x: ascii array
        :type x: numpy array
        :param pair: list of randomly chosen columns in the aschii matric, defaults to [0, 0]
        :type pair: list, optional
        :param mutate_threshold: residue in column will mutate if random number exceeds threshold, defaults to 0.8
        :type mutate_threshold: float, optional
        :return: ascii matric with co-varying amino acid columns
        :rtype: numpy array
        """
        # Site-specific amino-acids
        permitted_contacts = ['L', 'A']

        # Specify the marginals for L and for A
        marginals = [1 - mutate_threshold, mutate_threshold]

        # Specify the permitted binding partners for  L and A
        compliment = {'A': ['D', 'I', 'S'],
                      'L': ['G', 'C', 'M']}

        # Proabilistically specify the residues in the base column
        alpha_tokens = random.choices(
            permitted_contacts, k=np.shape(x)[0], weights=marginals)
        ascii_tokens = [self.__char_to_ascii(x) for x in alpha_tokens]

        # Make character selection using residue bias for the co-varying column
        bias = [0.1, 0.3, 0.5]
        alpha_compliments = [random.choices(compliment[x], k=1, weights=bias)[
            0] for x in alpha_tokens]

        # Convert to ascii
        ascii_compliments = [self.__char_to_ascii(
            x) for x in alpha_compliments]

        # Generate the co-varying columns
        x[:, pair[0]] = ascii_tokens
        x[:, pair[1]] = ascii_compliments
        return x

    def __build_encoder(self):
        """ Build and instantiate the char to ascii encoder

        :return: the char to ascii lookup table
        :rtype: dict
        """
        enc = []
        for char in self.tokens:
            enc.append(ord(char))
        encoder = dict(zip(self.tokens, enc))
        return encoder

    def __build_decoder(self):
        """ Build and instantiate the ascii to char decoder

        :return: the ascii to char lookup table
        :rtype: dict
        """
        enc = []
        for char in self.tokens:
            enc.append(ord(char))
        decoder = dict(zip(enc, self.tokens))
        return decoder

    def get_anndata_field(self, anndata, field='STRING_ID1', pair_id=['A', 'B']):
        """Extracts a column from a CSV file based on column name at a specific row

        :param field: column name, defaults to 'STRING_ID1'
        :type field: str, optional
        :param pair_id: the two protein identifiers for the specific file, defaults to [0, 0]
        :type pair_id: list, optional
        :return: the subset data belonging to the column name given by "field"
        :rtype: Series
        """
        # Subset the field from the row specified by pair_id
        f = anndata.loc[(anndata['STRING_ID1'] == pair_id[0])
                        & (anndata['STRING_ID2'] == pair_id[1])][field]
        return f

    def enforce_coevolution_signal(self, msa, pair_id, anndata):

        # Convert MSA to Ascii
        ascii_msa = np.array(self.__encode_char_to_ascii_matrix(msa))

        # Get protein-A length
        plen1 = int(self.get_anndata_field(
            anndata=anndata, field='len1', pair_id=pair_id))

        # Get protein-B length
        plen2 = int(self.get_anndata_field(
            anndata=anndata, field='len2', pair_id=pair_id))

        print("protein lengths")
        print(plen1, plen2)

        # Fix residues to A in protein-A regions of MSA
        n = 1
        pair_1 = random.sample(range(0, plen1), n)[0]
        pair_2 = random.sample(range(plen1 + 1, plen2), n)[0]
        pair = (pair_1, pair_2)

        print(pair)

        # Enfoce correlated signal
        ascii_corr_msa = self.__generate_covariate_pair(
            x=ascii_msa, pair=pair, mutate_threshold=0.8)

        # Reconvert MSA to alpha-numeric
        alpha_corr_msa = self.__decode_ascii_matrix_to_char_matrix(
            ascii_corr_msa)

        # Return MSA as alignment object
        alignment_msa = self.convert_msa_to_alignment(alpha_corr_msa)
        return alignment_msa


class ProteinParser:
    def __init__(self, anndata_path, paired_msa_path):

        # Initialize variables
        self.anndata_path = anndata_path
        self.paired_msa_path = paired_msa_path

    def extract_neighbourhood(self, df, top_pairs):
        frag_dict = {}
        df_copy = df.copy()

        for idx in df_copy.index:
            try:
                # Get obs
                obs = df_copy.iloc[idx, :]
                top_pair = obs.inter_dca_pairs[0]
                top_pair = [int(x) for x in top_pair]
                top_dca = obs.inter_dca_scores[0]

                # Get the protein lengths
                len_1 = df_copy.iloc[idx, :].len1
                len_2 = df_copy.iloc[idx, :].len2

                # Get pair id
                pair_id = [df_copy.iloc[idx, :].STRING_ID1,
                           df_copy.iloc[idx, :].STRING_ID2]

                # Extract flanks
                n_flanking = 5
                top_residues = top_pairs[0]
                smi_1, smi_2 = self.extract_flanking_residues(top_residues=top_residues,
                                                              pair_id=pair_id, plens=[
                                                                  len_1, len_2],
                                                              n_flanking=n_flanking)

                frag_dict[obs.id] = [top_dca, top_pair,
                                     n_flanking, smi_1, smi_2]
            except:
                pass

        return frag_dict

    def extract_flanking_residues(self,
                                  pair_id,
                                  top_residues,
                                  plens,
                                  n_flanking):
        """Get the residues positions n steps to the 
        left, and right of the top scoring DCA residue.

        :param anndata_path: path to the annotation dataframe
        :type anndata_path: string
        :param paired_msa_path: path to the paired msa sets
        :type paired_msa_path: string
        :param pair_id: string conraininf protein ID's in the form 'xandy'
        :type pair_id: string
        :param top_residues: top-1 DCA interacting residue pair integers
        :type top_residues: list
        :param plens: list containing lengths of of both proteins
        :type plens: list
        :param n_flanking: integer specifying the number of flanking residues
        :type n_flanking: int
        :return: The SMIELS string containing the the subset neighborhood pf reidues around top-1 DCA pair
        :rtype: string
        """

        flanks = []
        dl = DcaLab(anndata_path=self.anndata_path)

        for i in range(0, 2):
            # Extract the flanking integers
            pos_left = top_residues[i]
            pos_right = top_residues[i]

            # Instantiate empty lists
            right_flank = []
            left_flank = []
            for _ in range(0, n_flanking):
                pos_right += 1
                pos_left -= 1

                # Append residues n-steps to the right of top dca scoring pair
                if pos_right >= int(top_residues[i]) and ~(pos_right > int(plens[i])):
                    right_flank.append(pos_right)

                # Append residues n-steps to the left of top dca scoring pair
                if pos_left <= int(top_residues[i]) and pos_left >= 0:
                    left_flank.append(pos_left)

            # Flip the oder of the left flank
            left_flank = left_flank[::-1]
            flanks[i] = left_flank + [top_residues[i]] + right_flank

        # The next part of the functions
        ids = "and".join([pair_id[0], pair_id[1]])
        dca_file = ids + ".fasta"
        msa_file_path = os.path.join(self.paired_msa_path, dca_file)
        coupler = dl.generate_frequency_coupler(
            msa_file_path=msa_file_path)
        msa = coupler.convert_msa_to_df()
        msa = msa.iloc[0, :]

        # Extract protein regions
        p1 = "".join(msa.iloc[flanks[0]].values)
        p2 = "".join(msa.iloc[flanks[1]].values)
        return p1, p2


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


def get_structure(pdbfile, pdbid="system"):

    # convenience functions
    sloppyparser = Bio.PDB.PDBParser(
        PERMISSIVE=True, structure_builder=SloppyStructureBuilder()
    )

    return sloppyparser.get_structure(pdbid, pdbfile)


def get_dca_stats(df):
    """This function will iterate through every row of a pandas dataframe and return the DCA and frequency matrix stats.


    :param df: Annotation dataframe (anndata) object
    :type df: df
    :return: Collection of states on DCA AND Frequency matrix: top_dca_scores, frequency_variances, and frequeny ranks.
    :rtype: list
    """

    # Paths
    anndata_path = '/mnt/mnemo1/sum02dean/dean_mnt/projects/ecoli/annotation/all_ppi_info_frame.csv'
    paired_msa_path = '/mnt/mnemo1/sum02dean/dean_mnt/projects/ecoli/paired_msa'
    cache_dir_path = '/mnt/mnemo1/sum02dean/dean_mnt/projects/configs'

    # Initialize empty containers
    top_dca_scores = []
    frequency_variances = []
    ranks = []
    flat_freqs = []
    id_list = []

    for x in df.values:
        # Get the file name based on STRING ID
        ids = "and".join([x[0], x[1]])
        id_list.append(ids)
        dca_file = ids + ".fasta"
        msa_file_path = os.path.join(paired_msa_path, dca_file)

        # Exrtact DCA score
        dl = DcaLab(anndata_path=anndata_path)
        coupler = dl.generate_frequency_coupler(msa_file_path=msa_file_path,
                                                cache_dir_path=cache_dir_path)

        # Return the DCA scores if ppre-computed
        inter_dca, _ = coupler.get_dca_scores()

        # Subest the pairs
        dca_score = inter_dca[0, -1]
        pairs = inter_dca[:, 0:2]
        top_pair = coupler.subset_top_dca_couplings(pairs[0, 0:2])

        # Compute the frequency matrix
        freq_mat = coupler.get_frequency_matrix(top_pair)
        mat_rank = matrix_rank(freq_mat)
        flattened = freq_mat.flatten()
        mat_var = np.var(flattened[flattened > 0])

        # Collect stats
        top_dca_scores.append(dca_score)
        ranks.append(mat_rank)
        frequency_variances.append(mat_var)
        flat_freqs.append(flattened)

    stats = pd.DataFrame.from_dict({'id': id_list, 'dca': top_dca_scores, 'ranks': ranks,
                                    'variance': frequency_variances, 'frequencies': flat_freqs})

    return stats


def get_dca_stats_with_figs(df, save_fig=True):
    """This function will iterate through every row of a pandas dataframe and return the DCA and frequency matrix stats.


    :param df: Annotation dataframe (anndata) object
    :type df: df
    :return: Collection of stats on DCA and Frequency matrix: top_dca_scores, frequency_variances, and frequeny ranks.
    :rtype: list
    """

    # Paths
    anndata_path = '/mnt/mnemo1/sum02dean/dean_mnt/projects/ecoli/annotation/all_ppi_info_frame.csv'
    paired_msa_path = '/mnt/mnemo1/sum02dean/dean_mnt/projects/ecoli/paired_msa'

    # Initialize empty containers
    top_dca_scores = []
    frequency_variances = []
    ranks = []
    flat_freqs = []
    id_list = []
    figs = []

    for x in df.values:
        # Get the file name based on STRING ID
        ids = "and".join([x[0], x[1]])
        id_list.append(ids)
        dca_file = ids + ".fasta"
        msa_file_path = os.path.join(paired_msa_path, dca_file)

        # Exrtact DCA score
        coupler = fc(msa_file_path=msa_file_path,
                     annotation_file_path=anndata_path)

        # Return the DCA scores if ppre-computed
        inter_dca, _ = coupler.get_dca_scores()

        # Subest the pairs
        dca_score = inter_dca[0, -1]
        pairs = inter_dca[:, 0:2]
        top_pair = coupler.subset_top_dca_couplings(pairs[0, 0:2])

        # Compute the frequency matrix
        freq_mat = coupler.get_frequency_matrix(top_pair)
        mat_rank = matrix_rank(freq_mat)
        flattened = freq_mat.flatten()
        mat_var = np.var(flattened[flattened > 0])

        # Collect stats
        top_dca_scores.append(dca_score)
        ranks.append(mat_rank)
        frequency_variances.append(mat_var)
        flat_freqs.append(flattened)

        if save_fig:
            f = coupler.visualize_frequency_matrix()
            figs.append(f)
        else:
            figs.append(0)

    stats = pd.DataFrame.from_dict({'id': id_list, 'dca': top_dca_scores, 'ranks': ranks,
                                    'variance': frequency_variances, 'frequencies': flat_freqs, 'figs': figs})

    return stats


def parallelize_dataframe(df, func, n_cores=20):
    """This script takes a function 'func' and applies it to a n_obs/n_cores chunks 
       of a pandas DataFrame, it was purpose built for the func get_dca_stats.py.

    :param df: pandas annotation dataframe for PPI all info
    :type df: pandas DataFrame
    :param func: function to parallelize
    :type func: object
    :param n_cores: number of cores to parallelize the process over, defaults to 8
    :type n_cores: int, optional
    :return: pandas DataFrame containing the concatenated outputs of get_dca_stats()
    :rtype: pandas DataFrame
    """

    # Split the pandas DataFrame into n jobs
    df_split = np.array_split(df, n_cores)
    p = Pool(n_cores)

    # Aggregate the outputs of the parallel pool as df
    y = pd.concat(p.map(func, df_split))
    y.reset_index(inplace=True, drop=True)

    # Close the process
    p.close()
    p.join()
    return y


def get_label(file, labels):
    pair_1 = file.split('/')[-1]
    pair_1, pair_2 = pair_1.split("and")
    pair_1 = pair_1.replace(".gpickle", "")
    pair_2 = pair_2.replace(".gpickle", "")
    l = int(labels.loc[(labels.protein_1 == pair_1)
            & (labels.protein_2 == pair_2)].label)
    return file, l


def read_graphs(file_set):
    g_list = []
    for i, file in enumerate(file_set):
        G = nx.read_gpickle(file)
        g_list.append(G)
    return g_list


def format_graphs(graphs, label=1):
    graph_list = []
    # Convert into pytorch geoetric dataset: Positive
    for i, x in enumerate(tqdm(graphs)):
        F = nx.convert_node_labels_to_integers(x)
        for (n1, n2, d) in F.edges(data=True):
            d.clear()
        data = convert.from_networkx(F, group_edge_attrs=None)
        data.y = torch.FloatTensor([label])
        graph_list.append(data)
    return graph_list


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc
