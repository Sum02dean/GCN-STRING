# Import libraries
import os
from rdkit.Chem.rdmolops import GetAdjacencyMatrix as Adj
from spektral.datasets import TUDataset
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import glob
import re


class GraphMaker:
    """GraphMaker is used to generate a bi-molecular graph given two FASTA SMILES strings - this is intended
       for building graph neural networks on protein-protein molecular strucutres.     
    """

    def __init__(self, smiles_a, smiles_b, n_flanking):
        """Initialization takes in the FASTA SMILES from protein A and protein B
           together with the number of flanking residues around the DCA bridge.

        :param smiles_a: FASTA SMILES for protein A fragment
        :type smiles_a: Str
        :param smiles_b: FASTA SMILES for protein B fragment
        :type smiles_b: Str
        :param n_flanking: The number of amino acids flanking the central amino acid
        :type n_flanking: int
        """
        self.n_flanking = n_flanking
        self.smiles_a = smiles_a
        self.smiles_b = smiles_b
        self.smiles = smiles_a + smiles_b
        mols = []

    def get_mols(self, get_independent=False):
        """ Return the Mol objects for the provided smiles strings 
           Can return fragment mols or bi-molecular mols.

        :param get_independent: Bool that determines whether to return protein fragment Mol objects or
                                or di-protein Mol object, defaults to False
        :type get_independent: bool, optional
        :return: Mol objects for di-protein SMILES  or Mol objects for protein fragments
        :rtype: list of Mol objects
        """

        if get_independent:
            s = self.smiles
        else:
            s = [self.smiles_a, self.smiles_b]
        mols = []
        for x in s:
            mol_x = AllChem.MolFromFASTA(x)
            mols.append(mol_x)
        return mols

    def get_xyz_block(self, mol):
        """Converts Mol object to an XYZ printout.

        :param mol: Mol Object
        :type mol: Object
        :return: Information of Molecular XYZ coordinates. 
        :rtype: Str
        """
        block = AllChem.rdmolfiles.MolToXYZBlock(mol)
        return block

    def mol_2_molx_file(self, mol, file_name):
        """Converts Mol object to *.XYZ file (molx) file

        :param mol: Mol object
        :type mol: Object
        :param file_name: Name of the file to create
        :type file_name: Str
        :return: XYZ block - saves information to file
        :rtype: Str
        """

        # Save to mol format
        fn = os.path.join('mol_files', file_name.lower() + '.mol')
        Chem.rdmolfiles.MolToMolFile(mol, fn)

        # Save to molx format
        AllChem.EmbedMolecule(mol)
        fn = os.path.join('mol_files', file_name.lower() + '.xyz')
        AllChem.MolToXYZFile(mol, fn)
        AllChem.EmbedMolecule(mol)
        xyz_block = AllChem.rdmolfiles.MolToXYZBlock(mol)
        return xyz_block

    def get_atom_names(self, mol):
        """ Get the names of the individual atoms of the provided Mol object
            in canonical order.

        :param mol: Mol object
        :type mol: Object
        :return: Returns a list of the atoms as chemical symbols
        :rtype: list of element type Str
        """
        symbol = []
        for x in mol.GetAtoms():
            at = x.GetSymbol()
            symbol.append(at)
        return symbol

    def get_adj_matrices(self, mols):
        """Returns the adjacency matrix of the Mol Object.

        :param mols: Mol object to generate adjacency matrix from
        :type mols: Object
        :return: Binary adjacency matrix
        :rtype: Numpy array
        """
        adjs = []
        for x in mols:
            adj_x = Adj(x)
            adjs.append(adj_x)
        return adjs

    def get_adj_lengths(self):
        """ Reports the shape of the adjacency matrix for each protein fragmebt.

        :return: a list containing the size of each frament adjacency  matrix       
        :rtype: list of ints
        """
        lengths = []
        mols = self.get_mols(get_independent=True)
        for x in mols:
            lengths.append(len(Adj(x)))
        return lengths

    def combine_adj_matrices(self, adjs):
        # Build combined adjacency matrix
        max_len = len(adjs[0]) + len(adjs[1])
        combined_dca = np.zeros((max_len, max_len), dtype=int)

        # Populate with adjacecny_matrix A
        combined_dca[0:len(adjs[0]),
                     0:len(adjs[0])] = adjs[0]

        # Populate with adjacecny_matrix B
        combined_dca[len(adjs[0]): len(adjs[0]) + len(adjs[1]),
                     len(adjs[0]):len(adjs[0]) + len(adjs[1])] = adjs[1]
        return combined_dca

    def show_graph(self, adj, node_size=50, show_labels=False, labels_dict=None):
        """ Displays the network-x graph structure based on adjaceny matrix.

        :param adj: Adjacency matric of the Mol object
        :type adj: numpy arrau
        :param node_size: size of the node for visulisation purposes, defaults to 50
        :type node_size: int, optional
        :param show_labels: If True, shows the node labels, defaults to False
        :type show_labels: bool, optional
        :param labels_dict: Provide node labels as a dictionary key is int, value is string, defaults to None
        :type labels_dict: dict, optional
        :return: graph object with visulisation 
        :rtype: network-x graph object
        """
        rows, cols = np.where(adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        nx.Graph
        gr.add_edges_from(edges)
        #pos = nx.spring_layout(gr, k=k)
        if labels_dict:
            nx.draw(gr, with_labels=show_labels,
                    labels=labels_dict, node_size=node_size)
        else:
            nx.draw(gr, with_labels=show_labels, node_size=node_size)
        return gr

    def get_mol_with_atom_index(self, mol):
        " Dsiplay molecular with atom numbers"
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        return mol

    def get_adj_df(self, adj, atom_labels):
        """Get the adjacecny matrix as a pandas DataFrame and label rows and columns with atom symbols

        :param adj: adjacency matrix of Mol object
        :type adj: numpy array
        :param atom_labels: string names of the atoms
        :type atom_labels: Str
        """
        adJ = pd.DataFrame(adj)
        adJ.index = atom_labels
        adJ.columns = atom_labels

    def get_edge_list(self, adj):
        """ Get the edge identifiers from adjacency matrix  

        :param adj: adjacency matrix of Mol object
        :type adj: numpy array
        :return: list of edge labels as node-node id's
        :rtype: list of strings
        """
        jg = nx.from_numpy_matrix(adj)
        lg = nx.line_graph(jg)
        eg = nx.adj_matrix(lg)
        arr = eg.todense()
        gr = self.show_graph(adj, show_labels=True)
        el = gr.edges
        return el

    def get_ordered_bonds(self, mol, edge_list):
        """ Returns a list of edges ordered in a canoncialixed way.

        :param mol: Mol object to generate edge_list from
        :type mol: Mol object
        :param edge_list: The edge list extracted using get_edge_list()
        :type edge_list: list
        :return: list of sotred strings
        :rtype: list of strings
        """

        bonds = []
        atoms_ids = []

        # Make sure that the edge list is ordered and subet
        for i, (atom_1, atom_2) in enumerate(edge_list):
            bond_types = str(mol.GetBondBetweenAtoms(
                atom_1, atom_2).GetBondType())
            bonds.append(bond_types.split('.')[-1])
            atoms_ids.append((atom_1, atom_2))
        return bonds, atoms_ids

    def extract_kallisto(self, feature='vdw'):
        """ Calculates tha atomic properties using a call to the command line tool Kallisto.

        :param feature: van der waals radii ('wdv') or partial charges ('eeq'), defaults to 'vdw'
        :type feature: str, optional
        """
        # Create function to extract atomic descriptors (mol files need to be present)
        if feature == 'vdw':
            # Van der waals
            get_ipython().system(
                'bash /home/cluster/dsumne/data/MNF/src/scripts/extract_kallisto.sh vdw')

        elif feature == 'eeq':
            # Partial charges
            get_ipython().system(
                'bash /home/cluster/dsumne/data/MNF/src/scripts/extract_kallisto.sh eeq')
        return

    def create_bi_molecular_graph(self, nf_1=None, ef_1=None, nf_2=None, ef_2=None):
        """ Generates two graph objects from both protein fragments, populates their node /edge features and returns
            connected graphs on based on the central amino acid position in both protein fragments. 

        :param nf_1: node features for fragment 1, defaults to None
        :type nf_1: dictionary, optional
        :param ef_1: edge_features fragment 1, defaults to None
        :type ef_1: dictionary, optional
        :param nf_2: node features for fragment 2, defaults to None
        :type nf_2: dict, optional
        :param ef_2: edge_features fragment 2, defaults to None
        :type ef_2: dict, optional
        :return: conjoined bi-molecular network-x graph
        :rtype: network-x graph object
        """

        # Get the graphs for each fragment
        mol_1, mol_2 = self.get_mols()
        j1, j2 = self.get_adj_matrices([mol_1, mol_2])

        # Instantiate empty graphs
        gr_1 = nx.Graph()
        gr_2 = nx.Graph()

        # Extract node indices from adjacency matrices
        nodes_1 = [x for x in range(0, len(j1))]
        nodes_2 = [x for x in range(0, len(j2))]

        # Get edge list from adjacency matrices
        el_1 = self.get_edge_list(j1)
        el_2 = self.get_edge_list(j2)

        # Get all moleules without peptide reaction
        mols = self.get_mols(get_independent=True)

        # Get length of independent amino acids in full chain
        len_list = []
        for mol in mols:
            len_list.append(len([x for x in mol.GetAtoms()]))

        # Split over protein fragments
        len_1 = len_list[:len(self.smiles_a)]
        len_2 = len_list[len(self.smiles_b):]

        # Get alpha-C idx from fragments
        i_1 = int(np.floor(len(self.smiles_a)//2))
        i_2 = int(np.floor(len(self.smiles_b)//2))

        # Offset for the peptide bond formation and 0'th index
        idx_1 = sum(len_1[:i_1]) + (self.n_flanking - 1)
        idx_2 = sum(len_2[:i_2]) + (self.n_flanking - 1)

        # Get edge labels and atom indices
        edge_labels_1, atom_ids_1 = self.get_ordered_bonds(mol_1, el_1)
        edge_labels_2, atom_ids_2 = self.get_ordered_bonds(mol_2, el_2)

        # Get edge feature vectors frag-1
        # Coordinates
        a_1 = [x[0] for x in el_1]
        b_1 = [x[1] for x in el_1]
        # Features
        c_1 = [{'edge_type': x} for x in edge_labels_1]
        # Edge args
        d_1 = tuple(zip(a_1, b_1, c_1))

        # Get edge feature vectors frag-2
        # Coordinates
        a_2 = [x[0] for x in el_2]
        b_2 = [x[1] for x in el_2]
        # Features
        c_2 = [{'edge_type': x} for x in edge_labels_2]
        # Edge args
        d_2 = tuple(zip(a_2, b_2, c_2))

        # Create graph nodes and edges
        gr_1.add_nodes_from(nf_1)
        gr_1.add_edges_from(ef_1)  # Remember to replace with ef_1
        gr_2.add_nodes_from(nf_2)
        gr_2.add_edges_from(ef_2)  # Remember to replace with ef_2

        # Add intergraph edge with DCA label
        U = nx.union(gr_1, gr_2, rename=('a-', 'b-'))
        U.add_edge('a-' + str(idx_1), 'b-' + str(idx_2), edge_type='DCA')

        # Get adjacecny matrix
        j = nx.adjacency_matrix(U)
        j = j.todense()
        nx.draw(U, with_labels=True)
        return U, j
