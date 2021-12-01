import os
from Bio import AlignIO
import seaborn as sns
import glob
from Bio.Align import AlignInfo
from Bio import SeqIO
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from pydca.meanfield_dca import meanfield_dca

""" This class is used to compute DCA scores on combined protein MSA .fasta files. 
    The frequency matix can also be computed alonside a visualization.
"""


class FrequencyCoupler:
    """Class for computing and visualizing the freuqency matrix of top-n DCA
       aligned inter-protein residue pairs.
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
            if (pairs[0] <= plen1-1) and (pairs[1] > plen1-1):
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
            if (pairs[0] <= plen1-1) and (pairs[1] <= plen1-1):
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
            if (pairs[0] > plen1-1) and (pairs[1] > plen1-1):
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
                keys = token_a+token_b
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
        coupling_matrix = coupling_matrix/msa_depth
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
