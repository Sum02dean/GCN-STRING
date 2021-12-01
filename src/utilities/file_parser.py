import glob
import os
import sys
import shutil

""" DESCRIPTION:
    This is a class designed to parse the blastp file directory found in Taos remote
    location "/mnt/mnemo6/tao/PPI_Coevolution/ecoli_size2complex_new/blastp".
    The aim is to take any file matching a specific pattern and to collate it into a list.
    These files can then be moved or copied en bulk to a new location.
"""


class FileParser:
    """File parser to grab co-evolution, entropy and annotation data for protein pairs.
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


""" EXAMPLE:
    FP = FileParser(root_path='../../../../blastp')
    coevo_data = FP.get_data(category='coevolution')
    print(coevo_data[1:5])
    FP.copy_data(coevo_data[1:5], dest_dir='../../../../ecoli/coevolution')
"""
