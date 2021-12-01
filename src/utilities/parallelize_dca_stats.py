import sys
import os  # nopep8
import numpy as np
sys.path.append("../../")  # nopep8
import pandas as pd
from multiprocessing import Pool


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

    # Split the pandas DataFrame into n=n_cores jobs
    df_split = np.array_split(df, n_cores)
    p = Pool(n_cores)

    # Aggreate the outputs of the parallel pool as df
    y = pd.concat(p.map(func, df_split))
    y.reset_index(inplace=True, drop=True)

    # Clse the process
    p.close()
    p.join()
    return y
