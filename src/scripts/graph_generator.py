import sys
import pandas as pd
sys.path.append("../")  # nopep8
from utilities.gcn_utills import *


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

# Netsurf paths
nsp_msa = "/mnt/mnemo6/tao/PPI_Coevolution/STRING_data_11.5/511145_netsurfp2_output/_EggNOGmaxLevel1224_map2MSA"
nsp_seq = "/mnt/mnemo6/tao/PPI_Coevolution/STRING_data_11.5/511145_netsurfp2_output/"


# n_samples: use -1 for all
n_samples = 10
GM = GraphMaker(anndata_path=anndata_path)
graphs, labels = GM.get_computational_graph(n_samples=n_samples, string_to_af=ecoli_ext_string_to_af,
                                            ns_path_msa=nsp_msa,
                                            ns_path_seq=nsp_seq,
                                            coevo_path=coevo_path,
                                            show=False)
