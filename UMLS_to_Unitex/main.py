import re
import sys
import os
from convert_to_unitex import umls_to_unitex
from create_homonyms import create_homonyms
from add_main_type import create_categories
from compress_homonyms import format_homonyms

# NOTE: Due to size of the files, this program peaks at around 11GB of memory usage 
def main():
    # Point these to your MRCONSO and MRSTY files
    conso_path = '/home/keith/Documents/Paea Intern/UMLS-2020AA_Full_Install/2020AA/META/MRCONSO.RRF'
    types_path = '/home/keith/Documents/Paea Intern/UMLS-2020AA_Full_Install/2020AA/META/MRSTY.RRF'

    # Combine MRCONSO and MRSTY to create a unitex-style dictionary
    umls_to_unitex(conso_path, types_path, 'umls.dic')

    # Combine homonyms in dictionary into one entry per homonym
    create_homonyms('umls.dic', 'homonyms.dic')

    # Assign each entry a category and separate into files
    create_categories('homonyms.dic', '')

    # The names of the produced files
    output_files = [
        'uncompressed_devices.dic',
        'uncompressed_drugs.dic',
        'uncompressed_disorders.dic',
        'uncompressed_procedures.dic'
    ]

    # Put homonyms into correct formatting
    format_homonyms(output_files, 'Categorized Dictionaries')

    # Remove intermediate files
    os.remove('umls.dic')
    os.remove('homonyms.dic')
    for file in output_files:
        os.remove(file)

if __name__ == '__main__':
    main()
