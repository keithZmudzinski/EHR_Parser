import re
import sys
import os
from convert_to_unitex import umls_to_unitex
from create_homonyms import create_homonyms
from create_dlc_dlf import create_dlc_dlf

def main():
    # Point these to your MRCONSO and MRSTY files
    conso_path = '/home/keith/Documents/Paea Intern/UMLS-2020AA_Full_Install/2020AA/META/MRCONSO.RRF'
    types_path = '/home/keith/Documents/Paea Intern/UMLS-2020AA_Full_Install/2020AA/META/MRSTY.RRF'

    # Combine MRCONSO and MRSTY to create a unitex-style dictionary
    # umls_to_unitex(conso_path, types_path, 'umls.dic')

    # Combine homonyms in dictionary into one entry per homonym
    #   and separate into category files
    # create_homonyms('umls.dic', 'Categorized Dictionaries')

    # Combine complex words into dlc and simple words into dlf
    create_dlc_dlf('Categorized Dictionaries')

    # Remove intermediate files
    # os.remove('umls.dic')

    print('Done!')

if __name__ == '__main__':
    main()
