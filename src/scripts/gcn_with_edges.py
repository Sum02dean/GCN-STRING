import os
import sys
sys.path.append("../")
from utilities.gcn_utills import SloppyPDBIO, SloppyStructureBuilder, GraphMaker
import glob
import pandas as pd




if __name__ == '__main__':
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

    # Load data
    coevo_path = pd.read_csv(os.path.join(root, no_phyla), sep='\t', header=None)
    meta_data = pd.read_csv(os.path.join(root, meta), sep='\t', header=0)

    # Netsurf outputs
    netsurf = glob.glob(os.path.join(netsurf_path, "*.csv"))
    netsurf.sort()

    # Generate map between protein names and netsurfp file paths
    seq_names = [x.split("/")[-1].replace(".csv", "") for x in netsurf]
    netsurf_d = dict(zip(seq_names, netsurf))

    # Load metadata
    meta_data = pd.read_csv(os.path.join(root, meta), sep='\t')

    # Instantiate the graph maker class and genetate graphs
    GM = GraphMaker(anndata_path=os.path.join(root, meta))
    GM.get_computational_graph(string_to_af, coevo_path, netsurf_d=netsurf_d, n_samples=-1)

    