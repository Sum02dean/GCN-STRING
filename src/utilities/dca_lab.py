import sys
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy
from ..utilities.frequency_coupler import FrequencyCoupler  # nopep8
from ..utilities.msa_synthesizer import MsaSynthesizer  # nopep8


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
        dca_array = np.zeros((plen_1+plen_2, plen_1+plen_2))
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
        plt.xticks(np.arange(0, L1+L2, step=20), fontsize=1)
        plt.yticks(np.arange(0, L1+L2, step=20), fontsize=1)

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
