import requests
import pprint
import sys
from urllib.error import HTTPError
import json
import time
from collections import Counter
from fuzzywuzzy import fuzz
import os
import re
import pandas as pd
import numpy as np
from string_grouper import match_strings, match_most_similar, group_similar_strings, StringGrouper
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

base_url = 'http://localhost:8020/ehrp/'
extract_url = 'extract'
lookup_url = 'lookup'
file = 'pitt.txt'
group_types = ""

def preprocess_json(json):
    resp_json = json
    return resp_json

def name_file(filePath, end='.txt'):
    filePath = filePath[:-4] + '_' + group_types + end
    print(filePath)
    return filePath

def create_sorted_labels(resp_json):
    f_time = time.time()
    sorted_labels = []
    with open(name_file('key_words_sorted.txt'), 'w') as outfile:
        labels = Counter(k['label'].lower() for k in resp_json[0]['instances'] if k.get('label'))
        for label, count in labels.most_common():
            f_label = label + "," + str(count) + "\n"
            outfile.write(f_label)
            sorted_labels.append(label)
    print('Time taken to get sorted labels:', time.time()-f_time)
    return sorted_labels

#creates a corpus that is cleaned of key words
def build_potential_corpus(corpus, sorted_labels):

    # text - medical words = potentially medical words
    labels = set(sorted_labels)
    with open(corpus, 'r') as infile:
        text = infile.read()

    resultWords  = [word for word in re.split("[^a-zA-Z0-9_-]+",text) if word not in labels]
    if resultWords is not None:
        resultWords = ' '.join(resultWords)
    file_name = name_file('no_key_word.txt')
    with open(file_name, 'w') as outfile:
        outfile.write(resultWords)
    
    return file_name

def get_label_length_df(sorted_labels):
    labels_df =  pd.DataFrame(sorted_labels, columns =['label'])
    labels_df['len'] = labels_df['label'].apply(lambda x: len(x.split()))
    return labels_df

def split_n_grams_s(text, N):
    text = text.split()
    grams = [text[i:i+N] for i in range(len(text)-N+1)]
    gram_list = []
    for gram in grams:
        gram = ' '.join(gram)
        gram_list.append(gram)

    grams_s =  pd.Series(gram_list)
    return grams_s

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def catch_misspellings(corpus, sorted_labels, ratio):
    f_time = time.time()
    corpus_clean = build_potential_corpus(corpus, sorted_labels)
    with open(corpus_clean, 'r') as infile:
        text = infile.read()
    labels = get_label_length_df(sorted_labels)
    text =  split_n_grams_s(text, 2)

    # USING sudo pip install string-grouper
    labels = pd.Series(labels['label'])
    # Create all matches:
    matches = match_most_similar(text, labels)
    # Display the results:
    df = pd.DataFrame({'labels': labels, 'misspellings': matches})
    df = df.query("labels != misspellings")
    df.to_csv('test_matches_drug.csv')
    print('Time taken to find misspellings:', time.time()-f_time)

def extract(corpus, types=[]):
    args = {}

    # If specific types requested
    if types:
        args['types'] = types

    # Use the text above
    if corpus == 'text':
        args['text'] = text
        query_time = time.time()
        resp = requests.post(base_url+extract_url, data=args)
    # Use pitt.txt file
    else:
        file = open(corpus, 'rb')
        query_time = time.time()
        resp = requests.post(base_url+extract_url,files={'file':file}, data=args)
        file.close()

    print('Time taken to get result:', time.time()-query_time)

    print('Query response:', resp.status_code)
    return resp

#run case example: python3 ./misspelling.py extract [data file location] [type]
#ulyana use case: python3 ./misspelling.py extract ./data/pitt-data_goodlines.txt drug

def main():
    args = sys.argv[1:]
    service = args[0]
    global group_types

    # Extract
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

        sorted_labels = create_sorted_labels(resp_json)
        groups = catch_misspellings(corpus, sorted_labels, 90)

    else:
        print('Incorrect service, use "extract"')
    

    try:
        resp.raise_for_status()
    except HTTPError as err:
        print(err)

if __name__ == '__main__':
    main()
