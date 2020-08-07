import requests
import pprint
import sys
from urllib.error import HTTPError
import json
import time
from collections import Counter
import os
import re
import pandas as pd
import numpy as np

base_url = 'http://localhost:8020/ehrp/'
extract_url = 'extract'
lookup_url = 'lookup'
file = 'pitt.txt'
group_types = ""

# Creates a "sorted_labels_[type].txt" file which contains single line entries of concept words
# format: LABEL,NUM WORDS,NUM CHARS (not including space),TF
# returns array of sorted_labels in order of decreasing TF
def create_sorted_labels(resp_json):
    f_time = time.time()
    sorted_labels = []
    with open(name_file('sorted_labels.txt'), 'w') as outfile:
        labels = Counter(k['label'].lower() for k in resp_json[0]['instances'] if k.get('label'))
        for label, count in labels.most_common():
            num_spaces = label.count(' ')
            f_label = label + "," + str(num_spaces + 1) + "," + str(len(label) - num_spaces) + "," + str(count) + "\n"
            outfile.write(f_label)
            sorted_labels.append(label)
    print('Time taken to get sorted labels:', time.time()-f_time)
    return sorted_labels

# Creates a "sorted_labels_dic_[type].txt" file which contains json dictionary of concept labels
# returns a dictionary object where...
# label : concept label found in ontology
# words : number of words in concept (i.e. "red blood cell" = 3
# chars : number of characters in concept not counting spaces (i.e. "red blood cell" = 12)
# tf    : term frequency
# in the format...

    # {
    #     label: concept,
    #     instances: [
    #         {
    #             words: '',
    #             chars: '',
    #             tf: '',
    #         }
    #     ]
    # }
def create_sorted_labels_dic(resp_json):
    f_time = time.time()
    label_dic = {}

    labels = Counter(k['label'].lower() for k in resp_json[0]['instances'] if k.get('label'))
    for label, count in labels.most_common():
        num_spaces = label.count(' ')
        label_dic[label] = (({
            'words':  (num_spaces + 1),
            'chars': (len(label) - num_spaces),
            'tf': (count),

        }))
    
    with open(name_file('sorted_labels_dic.txt'), 'w') as outfile:
        json.dump(label_dic, outfile)
    
    print('Time taken to get sorted labels:', time.time()-f_time)
    return label_dic

# Creates a "concordance_[type].txt" file which contains a list of contexts for each concept
# returns a dictionary object where...
# label     : concept label found in ontology
# context   : the left and right context in which term was found in
# ontology  : which ontology the concept definition was found in
# umid      : UMID of concept term
# correct   : flag for misspelling ML algorithm, always set to 0
# in the format...

    # {
    #     label: concept,
    #     instances: [
    #         {
    #             context: []'',
    #             ontology: []'',
    #             umid: []'',
    #             correct: []'',
    #         }
    #     ]
    # }

# TODO: check to make sure len(words) in concordance is equal to sorted_list
# TODO: fix for easier concordance creation concordance[label] = ...
def create_concordance(resp_json):
    f_time = time.time()
    conc={}

    for i in resp_json[0]['instances']:
        label = i['label'].lower()
        if(not conc.get(label)):
            conc[label] = []
        conc.get(label).append(({
                'context': i['context'],
                'ontology': i['onto'],
                'umid': i['umid'],
                'correct': 0,
        }))

    with open(name_file('concordance.txt'), 'w') as outfile:
        json.dump(conc, outfile)
    print('Time taken to build concordance:', time.time()-f_time)
    return conc

# Optional pretty print for terminal concordance printing
def pretty_print(conc, sorted_labels):
    n = int(input("Number top words? "))
    sorted_labels = sorted_labels[:n]
    show_context = input("Show context?(y/n)")
    if(show_context == "n"):
        print("\n----Most occuring labels----")
        for label in sorted_labels:
            print("LABEL: ", label)
    elif(show_context == "y"):
        print("\n----Most occuring labels and their context----")
        for label in sorted_labels:
            print("LABEL: ", label)
            for c in conc[label]:
                print("context: ", c['context'])
            print("\n")

# json pre-processing
# TODO: complete pre-processing step
def preprocess_json(json):
    resp_json = json
    return resp_json

# file naming conventions based on type
def name_file(filePath, end='.txt'):
    filePath = filePath[:-4] + '_' + group_types + end
    print(filePath)
    return filePath

# extract from flask
def extract(corpus, types=[]):
    args = {}

    # If specific types requested
    if types:
        args['types'] = types

    file = open(corpus, 'rb')
    query_time = time.time()
    resp = requests.post(base_url+extract_url,files={'file':file}, data=args)
    print('Query response:', resp.status_code)
    print('Time taken to get result:', time.time()-query_time)
    file.close()
    return resp

# TODO: test with multiple type creations
# TODO: combine with seperate_free_text.py
def main():
    args = sys.argv[1:]
    service = args[0]
    global group_types

    # Extract only
    if service == 'extract':
        # Either text or pitt.txt
        corpus = args[1]
        print("CORPUS: ", corpus)
        types = []

        if len(args) > 2:
            types = args[2:]
            group_types  = '_'.join(types)
            print("Types: ", types)
            print("Group Types: ", group_types)

        resp = extract(corpus, types)
        resp_json = preprocess_json(resp.json())

        create_concordance(resp_json)
        # create_sorted_labels(resp_json)
        # create_sorted_labels_dic(resp_json)
    else:
        print("Only extract may be called\n $python3 concordance.py extract [DATA] type")

    # Check for errors
    try:
        resp.raise_for_status()
    except HTTPError as err:
        print(err)

if __name__ == '__main__':
    main()
