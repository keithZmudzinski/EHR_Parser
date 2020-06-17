'''File for processing the MEDDRA dataset to create the dictionary'''

from __future__ import print_function
import os.path
import argparse
import re

def clean_str(line):
    '''Cleans string'''
    line = re.sub(r'[^0-9a-zA-Z-+]+', ' ', line)
    line = re.sub(r'\s+', ' ', line)
    # Keep Abbreviations
    mod_line = ""
    for word in line.split():
        mod_line += " " + (word.lower() if word != word.upper() else word)
    return mod_line.strip()

def get_lines(filename):
    '''Get formatted lines from meddra files'''
    umids = []
    labels = []
    with open(filename, 'r') as m_file:
        lines = m_file.readlines()
        for line in lines:
            umids.append(line.split('$')[0])
            labels.append(clean_str(line.split('$')[1]))
    return umids, labels

def parse_meddra(args):
    '''Method to parse meddra files'''
    rfile = open(args.output_file, 'w')

    # SOC
    umids, labels = get_lines(args.meddra_dir + "/soc.asc")
    for index, umid in enumerate(umids):
        entry = labels[index] + ',' + umid + '.N+DISORDER+MEDDRA+SOC'
        print(entry, file=rfile)


    # HLGT
    umids, labels = get_lines(args.meddra_dir + "/hlgt.asc")
    for index, umid in enumerate(umids):
        entry = labels[index] + ',' + umid + '.N+DISORDER+MEDDRA+HLGT'
        print(entry, file=rfile)

    # HLT
    umids, labels = get_lines(args.meddra_dir + "/hlt.asc")
    for index, umid in enumerate(umids):
        entry = labels[index] + ',' + umid + '.N+DISORDER+MEDDRA+HLT'
        print(entry, file=rfile)

    # PT
    umids, labels = get_lines(args.meddra_dir + "/pt.asc")
    for index, umid in enumerate(umids):
        entry = labels[index] + ',' + umid + '.N+DISORDER+MEDDRA+PT'
        print(entry, file=rfile)

    # LLT
    umids, labels = get_lines(args.meddra_dir + "/llt.asc")
    for index, umid in enumerate(umids):
        entry = labels[index] + ',' + umid + '.N+DISORDER+MEDDRA+LLT'
        print(entry, file=rfile)

    rfile.close()


def main():
    '''Main method : parse arguments and start API'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--meddra_dir', type=str, default='data/MedAscii',
                        help="Path to the meddra directory")
    parser.add_argument('--output_file', type=str, default='resources/meddra.dic',
                        help="Path to the output file location")
    args = parser.parse_args()
    parse_meddra(args)

if __name__ == '__main__':
    main()
