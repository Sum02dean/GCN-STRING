from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from Bio.Alphabet import SingleLetterAlphabet, Gapped, IUPAC
from Bio.Align import MultipleSeqAlignment
import numpy as np
import pandas as pd
import random

"""This class is responsible for artificially generating an MSA
     with some co-evolutionary contraints - used to test the pyDCA implimentation.
    """


class MsaSynthesizer:
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
        marginals = [1-mutate_threshold, mutate_threshold]

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
        pair_2 = random.sample(range(plen1+1, plen2), n)[0]
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
