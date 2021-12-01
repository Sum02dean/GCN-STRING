import os
from src.utilities.dca_lab import DcaLab  # nopep8


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
