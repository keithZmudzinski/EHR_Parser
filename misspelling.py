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
from string_grouper import match_most_similar
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

base_url = 'http://localhost:8020/ehrp/'
extract_url = 'extract'
lookup_url = 'lookup'
file = 'pitt.txt'
group_types = ""
text = "test"

def remove_file(filePath):
    if os.path.exists(filePath):
        print("Removing: \"", filePath, "\"")
        os.remove(filePath)
        return 1
    else:
        print("File: \"", filePath, "\" was not removed because it doesn't exist")
    return 0

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
    labels_df['len'] = labels_df['label'].apply(lambda x: len(re.split(r'[\s\-]', x)))
    return (labels_df, labels_df['len'].max())

def split_n_grams_s(text, N):
    text = text.split()
    grams = [text[i:i+N] for i in range(len(text)-N+1)]
    gram_list = []
    for gram in grams:
        gram = ' '.join(gram)
        gram_list.append(gram)

    grams_s =  pd.Series(gram_list)
    return grams_s
## cosine_faster
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
        


def catch_misspellings_cosine(corpus, sorted_labels, ratio):
    f_time = time.time()
    corpus_clean = build_potential_corpus(corpus, sorted_labels)
    with open(corpus_clean, 'r') as infile:
        text = infile.read()
    (labels, max_label_len) = get_label_length_df(sorted_labels)
    text = split_n_grams_s(text, max_label_len)
    labels = pd.Series(labels['label'])

    matches = match_most_similar(text, labels, regex='[,./]|\s', min_similarity=ratio)

    df = pd.DataFrame({'labels': labels, 'misspellings': matches})
    df = df.query("labels != misspellings") 
    df.to_csv('text_matches_drug.csv')

   # df_matches = pd.DataFrame(columns=['labels', 'misspelling'])
    # for i in range(2, max_label_len+2):
    #     text_ngrams =  split_n_grams_s(text, i)
    #     print(text_ngrams)
    #     q = "len == " + str(i - 1)
    #     print(q)
    #     label_ngrams = labels.query(q)
    #     print(label_ngrams.head())
    #     slabel_ngrams = pd.Series(label_ngrams['label'])
    #     match_ngrams = match_most_similar(text_ngrams, slabel_ngrams, regex='[,./]|\s', min_similarity=ratio)
    #     df_ngrams = pd.DataFrame({'labels': slabel_ngrams, 'misspelling': match_ngrams})
    #     df_ngrams = df_ngrams.query("labels != misspelling")
    #     f = 'test_matches_drug_' + str(i) + '.csv'
    #     df_ngrams.to_csv(f)
    
    
    
    
    # USING sudo pip install string-grouper
    # labels = pd.Series(labels['label'])
    # Create all matches:
    # matches = match_most_similar(text, labels)
    # Display the results:
    # df = pd.DataFrame({'labels': labels, 'misspellings': matches})
    # df = df.query("labels != misspellings")
    print('Time taken to find misspellings:', time.time()-f_time)

#  pip install python-Levenshtein
# catches misspellings in the dictionary, not in the text!
def catch_misspellings_levenshtein(sorted_labels, ratio):
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

def catch_misspellings_cosine_faster(corpus, sorted_labels, ratio):
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


    # # USING sudo pip install string-grouper
    # labels = pd.Series(labels['label'])
    # # Create all matches:
    # matches = match_most_similar(text, labels)
    # # Display the results:
    # df = pd.DataFrame({'labels': labels, 'misspellings': matches})
    
    
    # df = df.query("labels != misspellings")
    # df.to_csv('test_matches_drug.csv')
    # print('Time taken to find misspellings:', time.time()-f_time)

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
        catch_misspellings_cosine(corpus, sorted_labels, 0.8)
        remove_file('no_key_word_drug.txt')
        #catch_misspellings_levenshtein(sorted_labels, 90)

    else:
        print('Incorrect service, use "extract"')
    

    try:
        resp.raise_for_status()
    except HTTPError as err:
        print(err)

if __name__ == '__main__':
    main()
