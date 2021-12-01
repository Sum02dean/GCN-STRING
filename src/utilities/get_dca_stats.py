import sys
sys.path.append("../../")  # nopep8
import os
import numpy as np
from src.utilities.msa_synthesizer import MsaSynthesizer as MSA
from src.utilities.frequency_coupler import FrequencyCoupler as fc
from src.utilities.dca_lab import DcaLab
import pandas as pd
from numpy.linalg import matrix_rank


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
    :return: Collection of states on DCA AND Frequency matrix: top_dca_scores, frequency_variances, and frequeny ranks.
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
