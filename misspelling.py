import requests
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
from string_grouper import match_most_similar, match_strings
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

base_url = 'http://localhost:8020/ehrp/'
extract_url = 'extract'
lookup_url = 'lookup'
file = 'pitt.txt'
group_types = ""
text = "test"

# TODO: Implement pass through of create_sorted_labels from file or from concordance.py
# TODO: Speed up get_label_length
def create_sorted_labels(resp_json, file_bool):
    f_time = time.time()
    sorted_labels = []
    if (file_bool is True):
        with open(name_file('key_words_sorted.txt'), 'w') as outfile:
            labels = Counter(k['label'].lower() for k in resp_json[0]['instances'] if k.get('label'))
            for label, count in labels.most_common():
                num_spaces = label.count(' ')
                f_label = label + "," + str(num_spaces + 1) + "," + str(len(label) - num_spaces) + "," + str(count) + "\n"
                outfile.write(f_label)
                sorted_labels.append(label)
    else:
        labels = Counter(k['label'].lower() for k in resp_json[0]['instances'] if k.get('label'))
        for label, count in labels.most_common():
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

def get_matches(corpus, sorted_labels, ratio):
    f_time = time.time()
    corpus_clean = build_potential_corpus(corpus, sorted_labels)
    with open(corpus_clean, 'r') as infile:
        text_string = infile.read()

    Stext =  split_n_grams_s(text_string, 1)
    labels = get_labels_df(sorted_labels)
    labels = pd.Series(labels['label'])
    
    matches_df = get_cosim_matches(labels, Stext, ratio)
    #for now
    matches_df.to_csv(name_file('cosim_match.csv', '.csv'))

    matches_df = add_context(matches_df, text_string)
    

    print('Time taken to get matches:', time.time()-f_time)

# USING sudo pip install string-grouper
def get_cosim_matches(labels, text, ratio):
    print("\tMatches Using Cosine Similarity")
    m_time = time.time()
    matches_df = match_strings(labels, text, min_similarity=ratio)
    print('\tTime taken to cosim match all strings:', time.time()-m_time)
   # matches_df.to_csv('test_matches_ungroupped.csv')
    
    m_time = time.time()
    matches_df = matches_df.groupby(matches_df.columns.tolist()).size().reset_index().rename(columns={0:'freq','left_side':'label','right_side':'text','similarity':'cosim'})
    matches_df = matches_df.sort_values(by='cosim', ascending=False).reset_index().drop(columns=['index'])
    print('\tTime taken to group same matches:', time.time()-m_time)

    return matches_df

def get_context(word, text):
    print(word + text[0:20])

def add_context(matches_df, text):
    return matches_df

# Using pip install python-Levenshtein
# catches misspellings in the dictionary, not in the text!
def get_levsh_matches(sorted_labels, ratio):
    label_groups = list() # groups of names with distance > 80
    for label in sorted_labels:
        for group in label_groups:
            if all(fuzz.ratio(label, w) > ratio for w in group):
                group.append(label)
                break
        else:
            label_groups.append([label, ])
    
    n_groups = 0
    with open(name_file('catch_misspellings.txt'), 'w') as outfile:
        for label_group in label_groups:
            if(len(label_group) > 1):
                n_groups+=1
                for label in label_group:
                    outfile.write(label + ", ")
                outfile.write("\n")

    print("Number of groups in misspellings file: ", n_groups)
    
    return label_groups

# Utility for curent working code
def get_label_length_df(sorted_labels):
    labels_df =  pd.DataFrame(sorted_labels, columns =['label'])
    labels_df['len'] = labels_df['label'].apply(lambda x: len(re.split(r'[\s\-]', x)))
    return (labels_df, labels_df['len'].max())

def get_labels_df(sorted_labels):
    labels_df =  pd.DataFrame(sorted_labels, columns =['label'])
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

def remove_file(filePath):
    if os.path.exists(filePath):
        print("Removing: \"", filePath, "\"")
        os.remove(filePath)
        return 1
    else:
        print("File: \"", filePath, "\" was not removed because it doesn't exist")
    return 0

def name_file(filePath, end='.txt'):
    filePath = filePath[:-4] + '_' + group_types + end
    print(filePath)
    return filePath

def preprocess_json(json):
    resp_json = json
    return resp_json

#--------WIP---------------
# TODO: Implement coside faster by dissecting string-grouper class
def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def cosine_faster_csr(A, B, ntop, lower_bound=0):
    f_time = time.time()
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    print('Time taken to return csr matrix:', time.time()-f_time)

    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})

def get_cosim_matches_faster(corpus, sorted_labels, ratio):
    f_time = time.time()
    corpus_clean = build_potential_corpus(corpus, sorted_labels)
    with open(corpus_clean, 'r') as infile:
        text = infile.read()
    (labels_df, size) = get_label_length_df(sorted_labels)
    print(labels_df.head())
    # text =  split_n_grams_s(text, 2)
    # labels = labels_df['label']
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    # tf_idf_matrix_label = vectorizer.fit_transform(labels)
    # tf_idf_matrix_text = vectorizer.fit_transform(text)
    print(ngrams('today catheter'))
    
    labels = labels_df['label']
    v_time = time.time()
    tf_idf_matrix_label = vectorizer.fit_transform(labels)
    print('Time taken to vectorize labels:', time.time()-v_time)
    
    text =  split_n_grams_s(text, 2).tolist() #could be made more efficient
    v_time = time.time()
    tf_idf_matrix_text = vectorizer.fit_transform(text)
    print('Time taken to vectorize text-ngrams:', time.time()-v_time)
    matches = cosine_faster_csr(tf_idf_matrix_label, tf_idf_matrix_text, 10, 0.8)

    matches_df = get_matches_df(matches, labels, top=100000)
    matches_df = matches_df[matches_df['similairity'] < 0.99999] # Remove all exact matches
    print(matches_df.sample(20))

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

        sorted_labels = create_sorted_labels(resp_json, False)
        get_matches(corpus, sorted_labels, 0.8)
        # get_cosim_matches_faster(corpus, sorted_labels, 0.8)
        #get_levsh_matches(sorted_labels, 90)

        remove_file(name_file('no_key_word.txt'))

    else:
        print('Incorrect service, use "extract"')
    

    try:
        resp.raise_for_status()
    except HTTPError as err:
        print(err)

if __name__ == '__main__':
    main()
