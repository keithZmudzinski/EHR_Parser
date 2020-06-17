'''File for processing RXCONSO.RRF file from the UMLS to create the dictionary'''

from __future__ import print_function
import argparse
import re

def clean_str(line):
    '''Cleans string'''
    line = re.sub(r'[^0-9a-zA-Z-+\\().]+', ' ', line)
    line = re.sub(r'\.', '\\\.', line)
    line = re.sub(r'\s+', ' ', line)
    # Keep Abbreviations
    mod_line = ""
    for word in line.split():
        mod_line += " " + (word.lower() if word != word.upper() else word)
    return mod_line.strip()

def parse_rxnorm(args):
    '''Method to parse rxnorm files'''
    rfile = open(args.output_file, 'w')
    count = 0
    types = set()
    with open(args.rxnorm_dir + "/RXNCONSO.RRF") as rx_file:
        # otypes = ['DRUGBANK', 'GS', 'ATC', 'CVX', 'NDDF', 'MTHCMSFRF', 'MMSL', 'VANDF',
        #           'NDFRT', 'MSH', 'RXNORM', 'SNOMEDCT_US', 'MTHSPL', 'MMX']
        otypes = ['RXNORM']
        for line in rx_file:
            parts = line.split('|')
            if parts[11] not in types:
                types.add(parts[11])
            if len(parts) == 19 and parts[11] in otypes and parts[12] in ['IN', 'BN']:
                count += 1
                entry = clean_str(parts[14])+','+parts[0]+'.N+DRUG+'+parts[11]+"+"+parts[12]
                print(entry, file=rfile)
    print(count)
    print(types)
    rfile.close()

def main():
    '''Main method : parse arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxnorm_dir', type=str, default='data/rxnorm',
                        help="Path to the rxnorm directory")
    parser.add_argument('--output_file', type=str, default='resources/RxNorm.dic',
                        help="Path to the output file location")
    args = parser.parse_args()
    parse_rxnorm(args)

if __name__ == '__main__':
    main()
