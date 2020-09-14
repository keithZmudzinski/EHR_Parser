import requests
import sys
from urllib.error import HTTPError
import json
import time
from collections import Counter
from fuzzywuzzy import fuzz
import os
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from string_grouper import match_most_similar, match_strings
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import random
from pandas import json_normalize 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from gensim.models import Word2Vec, word2vec
import ast
from nltk.corpus import brown
import re
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

base_url = 'http://localhost:8020/ehrp/'
extract_url = 'extract'
lookup_url = 'lookup'
file = 'pitt.txt'
group_types = ""
text = "test"

pd.options.mode.chained_assignment = None

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
    text = get_text(corpus)
    
    resultWords  = [word for word in re.split("[^a-zA-Z0-9_-]+",text) if word not in labels]
    if resultWords is not None:
        resultWords = ' '.join(resultWords)
    file_name = name_file('no_key_word.txt')
    with open(file_name, 'w') as outfile:
        outfile.write(resultWords)
    
    return file_name

def get_matches(resp_json, corpus, sorted_labels, ratio=0.8, context_size=125, max_context=-1, context=True, group_labels=True):
    f_time = time.time()
    corpus_clean = build_potential_corpus(corpus, sorted_labels)
    text_string = get_text(corpus_clean)

    Stext =  split_n_grams_s(text_string, 1)
    labels = get_labels_df(sorted_labels)
    labels = pd.Series(labels['label'])
    
    matches_df = get_cosim_matches(labels, Stext, ratio, group_labels)
    #for now
    if (context == True):
        text = get_text(corpus)
        matches_df = add_context(matches_df, text, context_size)
        if(max_context > 0):
            matches_df = limit_context(matches_df, max_context)
    matches_df.to_csv(name_file('cosim_match.csv', '.csv'))

    print('Total time taken to get_matches:', time.time()-f_time)

# USING sudo pip install string-grouper
def get_cosim_matches(labels, text, ratio, group_labels):
    print("\tMatches Using Cosine Similarity")
    m_time = time.time()
    matches_df = match_strings(labels, text, min_similarity=ratio)
    print('\tTime taken to cosim match all strings:', time.time()-m_time)
   # matches_df.to_csv('test_matches_ungroupped.csv')
    if(group_labels == True):
        m_time = time.time()
        matches_df = matches_df.groupby(matches_df.columns.tolist()).size().reset_index().rename(columns={0:'freq_match','left_side':'label','right_side':'match','similarity':'cosim'})
        matches_df = matches_df.sort_values(by='cosim', ascending=False).reset_index().drop(columns=['index'])
        print('\tTime taken to group same matches:', time.time()-m_time)
    else:
        matches_df = matches_df.rename(columns={'left_side':'label','right_side':'match','similarity':'cosim'}).sort_values(by=['label','cosim'], ascending=False).reset_index().drop(columns=['index'])
    
    return matches_df


# TODO: Improve text chunks to consider left side + WORD + right side instead of randomn
# cuts out n-length character chunks from s

def get_text(file):
    with open(file, 'r') as infile:
        text = infile.read()
    return text

#up to 2890
def get_lines(file):
    with open(file) as infile:
        content = infile.read().splitlines()
    return content

# TODO: fix context to correct get
def get_context(word, text_chunks):
    contexts = [context + '.' for context in text_chunks if word in context]
    return contexts

def add_context(matches_df, text, chunk_n):
    f_time = time.time()
    text_chunks = chunks(text, chunk_n)
    matches_df['context'] = matches_df['label'].apply(get_context, args=[text_chunks])
    print('\tTime taken to get context:', time.time()-f_time)
    return matches_df

def limit_context(matches_df, max_context):
    matches_df['context'] = matches_df['context'].apply(lambda x: x if len(x) <= max_context else random.choices(x,k=max_context))
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

#TODO: make sure grams are not cut off mid-word
def split_n_grams_s(text, N):
    text = text.split()
    grams = [text[i:i+N] for i in range(len(text)-N+1)]
    gram_list = []
    for gram in grams:
        gram = ' '.join(gram)
        gram_list.append(gram)

    grams_s =  pd.Series(gram_list)
    return grams_s

# TODO Next: Implement skipgram into comparison vector
def get_skipgram(label, seq='odd'):
    if(seq == 'odd'):
        print("Odd skipgram not implemented")
    elif(seq == 'even'):
        print("Even skipgram not implemented")
    elif(seq == 'ddo'):
        print("Odd-backwards skipgram not implemented")
    elif(seq == 'neve'):
        print("Even-backwards skipgram not implemented")

#TODO: **difficult** figure out which chunk to add? same chunks keep appearing in file when True, False
def chunks(s, n):
    text_chunks = []
    for start in range(0, len(s), n):
        text_chunks.append(s[start:start+n])
    return text_chunks

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

def to_word_list(df_row, preprocess):
    sentence = df_row['context'].lower()
    label = str(df_row['label'])
    rx_label = "\\b[\S]*" + label + "[\S]*\\b"
    rx_dose = " [0-9]+ *[mg|g|m] *[g]* "
    rx_list = " [0-9]+[.|\)|\|] "
    rx_num = "\b[0-9]+\b"
    # print(rx_dose)
    # print(rx_num)
    if preprocess:
        # code below replaces all instances of "drug" with "drug" identifier: ulyana for word2vec
        if (df_row['is_drug'] == 1):
            # print(sentence)
            sentence = re.sub(rx_label, 'ulyana', sentence)
            sentence = re.sub(rx_dose, ' DOOSSEE ', sentence)
            sentence = re.sub(rx_num, ' LIISSTT ', sentence)
            sentence = re.sub(rx_num, ' NUUMM ', sentence)
            # print(sentence)
        # sentence = str(sentence).replace(label, "ulyana")
        sentence = re.sub("[^a-zA-Z0-9-]"," ", str(sentence))
        # convert to lower case and split at whitespace
        words = sentence.split()
        # remove stop words (false by default)
        # remove from stop words "on"
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        # sentence = " ".join(words)
        sentence = list(words)
    else:
        words = sentence.split()
        sentence = list(words)
    return sentence
    
def preprocess_exploded(df, preprocess=True):
    df['context'] = df.apply(to_word_list, axis=1, args=[preprocess])
    return df

#TODO: does not account for 'ulyana-'
#TODO: length of drug sentences too short, good for training bad for testing with context
def create_window(df_row, window):
    label = df_row['label_w2v']
    sentence = list(df_row['context'])
    index_label = -1
    if label in sentence:
        index_label = sentence.index(label)
    else:
        print('NO LABEL IN SENTENCE!!')
        print(sentence)
        print(label)
        return 0

    lwindow = index_label - window
    rwindow = index_label + window
    context = []
    if(lwindow >= 0 and rwindow < len(sentence)):
        context = sentence[lwindow:index_label] + sentence[index_label+1:rwindow+1]
    # if df_row['is_drug'] == 1:
        # print("index label: ", index_label, "window: ", window)
        # print("sentence: ", sentence)
        # print("context: ", context)
    if(len(context) == 0):
        return 0
    return context

# eval_guess = ['tp', 'fp', 'fn', 'tn', 0]
# tp = true positive where ulyana was guessed correctly
# fp = false positive where ulyana was guessed but correct answer ludmyla
# fn = false negative where ludmyla was guessed but correct answer ulyana
# tn = true negative where ludmyla was guessed correctly
# 0 is preset, where guess was not made (should we count it as tn?)
  
def guess_label(df_row, model):
    context = df_row['context_w2v']
    output_words = model.predict_output_word(context_words_list=context, topn=10000)
    if type(output_words) is not None:
        output_words_list = [i[0] for i in output_words or []]
        index_ulyana = output_words_list.index('ulyana') if 'ulyana' in output_words_list else 10009
        index_ludmyla = output_words_list.index('ludmyla') if 'ludmyla' in output_words_list else 10009
        if (index_ulyana < index_ludmyla):
            df_row['label_w2v_guess'] = 'ulyana'
            df_row['w2v_guess_p'] = output_words[index_ulyana][1]
            df_row['w2v_guess_index'] = index_ulyana
            df_row['eval_guess'] = 'tp' if df_row['label_w2v'] == 'ulyana' else 'fp'
            # return 'u'+ output_words[index_ulyana][1]
        elif (index_ulyana > index_ludmyla):
            df_row['label_w2v_guess'] = 'ludmyla'
            df_row['w2v_guess_p'] = output_words[index_ludmyla][1]
            df_row['w2v_guess_index'] = index_ludmyla
            df_row['eval_guess'] = 'tn' if df_row['label_w2v'] == 'ludmyla' else 'fn'
        else:
            #GUESS THAT if we can't mach ludmyla or ulyana with high positivity, match ludmyla
            df_row['label_w2v_guess'] = 'ludmyla'
            df_row['eval_guess'] = 'tn' if df_row['label_w2v'] == 'ludmyla' else 'fn'
            # return 'l'+ output_words[index_ludmyla][1]
    else:
        print("Reached Nonetype Suggestion")

    return df_row

def test_context_w2v(model, df, window=3):
    df['label_w2v'] = df['is_drug'].apply(lambda x: 'ulyana' if x == 1 else 'ludmyla')
    df['context_w2v'] = df.apply(create_window, axis=1, args=[window])
    print("Shape of testing batch: ", df.shape)
    df = df[df['context_w2v'] != 0]
    print("Shape of clean testing batch: ", df.shape)
    # df['label_w2v_guess'] = df.apply(guess_label, axis=1, args=[model])
    df['label_w2v_guess'] = 'guess'
    df['w2v_guess_p'] = 0
    df['w2v_guess_index'] = -1
    df['eval_guess'] = 0
    df = df.apply(guess_label, axis=1, args=[model])
    print(df)
    return df

#TODO expand example
def heuristics_word2vec(model):
    print("Model type: ", type(model))
    print("Word2Vec model heuristics: ", model)
    print("Similar to drug: ", model.wv.similar_by_word("ulyana", topn=5))
    print("Similarity: ulyana, today", model.wv.similarity("ulyana", "today"))
    print("Similarity: ulyana, daily", model.wv.similarity("ulyana", "daily"))
    print("Similarity: ulyana, tylenol", model.wv.similarity("ulyana", "tylenol"))
    print("Similar to non-drug: ", model.wv.similar_by_word("ludmyla", topn=5))
    print("Similarity: ludmyla, today", model.wv.similarity("ludmyla", "today"))
    print("Similarity: ludmyla, daily", model.wv.similarity("ludmyla", "daily"))
    print("Similarity: ludmyla, tylenol", model.wv.similarity("ludmyla", "tylenol"))
    f_time = time.time()
    print("Predicting top 5 likely words for \"Patient was prescribed _______ to take daily\"")
    print(model.predict_output_word(context_words_list=['patient', 'was', 'prescribed', 'to', \
        'take', 'daily'], topn=5))
    # print("Time taken to make top 5 assumption: ", time.time()-f_time)
    f_time = time.time()
    model.predict_output_word(context_words_list=['patient', 'was', 'prescribed', 'to', \
        'take', 'daily'], topn=100)
    # print("Time taken to make top 100 assumption: ", time.time()-f_time)
    f_time = time.time()
    output_words = model.predict_output_word(context_words_list=['patient', 'was', 'prescribed', 'to', \
        'take', 'daily'], topn=10000)
    # print(output_words[2][0])
    print("Time taken to make top 10000 assumption: ", time.time()-f_time)

def test_heuristics_w2v(df):
    print("MODEL HEURISTICS")
    metrics = df['eval_guess'].value_counts()
    tn = metrics['tn']
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(metrics)
    print("Precision: ", precision)
    print("Recall: ", recall)
    df_fp = df[df['eval_guess'] == 'fp']
    df_fn = df[df['eval_guess'] == 'fn']
    df_tn = df[df['eval_guess'] == 'tn']
    df_tp = df[df['eval_guess'] == 'tp']
    print("[Mean:Min:Max:Median] p certainty per guess type")
    print("tn\t", df_tn['w2v_guess_p'].mean(), "\t", df_tn['w2v_guess_p'].min(), "\t", df_tn['w2v_guess_p'].max(), "\t")
    print("tp\t", df_tp['w2v_guess_p'].mean(), "\t", df_tp['w2v_guess_p'].min(), "\t", df_tp['w2v_guess_p'].max(), "\t") 
    print("fp\t", df_fp['w2v_guess_p'].mean(), "\t", df_fp['w2v_guess_p'].min(), "\t", df_fp['w2v_guess_p'].max(), "\t") 
    print("fn\t", df_fn['w2v_guess_p'].mean(), "\t", df_fn['w2v_guess_p'].min(), "\t", df_fn['w2v_guess_p'].max(), "\t")
    print("tn full description: \n", df_tn['w2v_guess_p'].describe())
    print("tp full description: \n", df_tp['w2v_guess_p'].describe())
    print("fp full description: \n", df_fp['w2v_guess_p'].describe())
    print("fn full description: \n", df_fn['w2v_guess_p'].describe()) 

def train_word2vec_vocab(name,train_sentences):
    model_name = name
    # Set values for various word2vec parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 3   # Minimum word count                        
    num_workers = 3       # Number of threads to run in parallel
    context = 200          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    if not os.path.exists(model_name): 
        # Initialize and train the model (this will take some time)
        model = Word2Vec(train_sentences, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling, hs=1)

        # If you don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)
        model.save(model_name+".model")
    else:
        model = Word2Vec.load(model_name+".model")
    
    heuristics_word2vec(model)
    return (model, num_features)

def train_word2vec_context(model, df, train_sentences):
    model = word2vec.Word2Vec(sentences=train_sentences)

    # df['label_w2v'] = df['is_drug'].apply(lambda x: 'ulyana' if x == 1 else 'ludmyla')
    # print(df)
    # model = word2vec.train_cbow_pair(model, word, input_word_indices, l1, alpha, \
    #         learn_vectors=True, learn_hidden=True, compute_loss=False, \
    #         context_vectors=None, context_locks=None, is_ft=False)
    return 'hehe'

def rf_classifier(df_known_file, df_unknown_file, exploded=True, preprocess=True, context_size=60):
    # Load in data
    df = pd.read_csv(df_known_file)
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    # df['context'] = df['context'].apply(preprocess_to_wordlist)
    if exploded:
        df = preprocess_exploded(df, preprocess)
        print("Whole Dataframe size: ", df.shape)
        df_train = df.sample(n=int(df.shape[0]* 0.75))
        df_test = df[~df.isin(df_train)].dropna()
        df_train['context'] = df_train.apply(limit_context_size,axis=1,args=[context_size,'list'])
        train_sentences = df_train['context'].to_list()
        print(train_sentences[0])
        print(train_sentences[1])
        print("Training size: ", df_train.shape)
        print("\tnumber training sentences: ", len(train_sentences))
        print("Testing size: ", df_test.shape)
    else:
        train_topics = df['context'].to_list()
        train_sentences = preprocess_to_sentence(train_topics)
   
    (model, num_features) = train_word2vec_vocab('word2vec_drug', train_sentences)
    df_test = test_context_w2v(model, df_test, window=3)
    df_test.to_csv('w2v_drugs_test.csv')
    test_heuristics_w2v(df_test)
   # (model) = train_word2vec_context('drug_train_model_context', df_train)

#TODO: Fix random forest
    # # calculate average feature vectors for training and test sets
    # clean_train_reviews = []
    # for review in df_train['context']:
    #     clean_train_reviews.append(review)
    # trainDataVecs = get_avg_feature_vecs(clean_train_reviews, model, num_features)

    # nan_indices = list({x for x,y in np.argwhere(np.isnan(trainDataVecs))})
    # if len(nan_indices) > 0:
    #     print('Removing {:d} instances from test set.'.format(len(nan_indices)))
    #     trainDataVecs = np.delete(trainDataVecs, nan_indices, axis=0)
    #     df_train.drop(df_train.iloc[nan_indices, :].index, axis=0, inplace=True)
    #     assert trainDataVecs.shape[0] == len(df_train)

    # clean_test_reviews = []
    # for review in df_test['context']:
    #     clean_test_reviews.append(review)
    # testDataVecs = get_avg_feature_vecs(clean_test_reviews, model, num_features)

    # nan_indices = list({x for x,y in np.argwhere(np.isnan(testDataVecs))})
    # if len(nan_indices) > 0:
    #     print('Removing {:d} instances from test set.'.format(len(nan_indices)))
    #     testDataVecs = np.delete(testDataVecs, nan_indices, axis=0)
    #     df_test.drop(df_test.iloc[nan_indices, :].index, axis=0, inplace=True)
    #     assert testDataVecs.shape[0] == len(df_test)

    # forest = RandomForestClassifier(n_estimators = 100)

    # print("Fitting a random forest to labeled training data...")
    # forest = forest.fit(trainDataVecs, df_train['is_drug'])

    # print("Predicting labels for test data..")
    # result = forest.predict(testDataVecs)

    # print(classification_report(df_test['is_drug'], result))
    # probs = forest.predict_proba(testDataVecs)[:, 1]

    # fpr, tpr, _ = roc_curve(df_test['is_drug'], probs)
    # auc = roc_auc_score(df_test['is_drug'], probs)

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='AUC {:.3f}'.format(auc))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()


# def make_feature_vec(words, model, num_features):
#     """
#     Average the word vectors for a set of words
#     """
#     feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
#     nwords = 0.
#     index2word_set = set(model.wv.index2word)  # words known to the model
#     for word in words:
#         if word in index2word_set: 
#             nwords = nwords + 1.
#             feature_vec = np.add(feature_vec,model[word])

#     feature_vec = np.divide(feature_vec, nwords)
#     return feature_vec


# def get_avg_feature_vecs(reviews, model, num_features):
#     """
#     Calculate average feature vectors for all reviews
#     """
#     counter = 0
#     review_feature_vecs = np.zeros((len(reviews),num_features), dtype='float32')  # pre-initialize (for speed)
#     for review in reviews:
#         review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
#         counter+=1
#     return review_feature_vecs

def replace_label(df_row, key):
    sentence = df_row['context']
    sentence = ' '.join(sentence)
    label = str(df_row['label'])
    rx_label = "\\b[\S]*" + label + "[\S]*\\b"
    if (df_row['is_drug']) == 0:
        sentence = re.sub(rx_label, key, sentence)
    return sentence.split()

def rand_label(df_row):
    sentence = df_row['context']
    label = "null"
    if(sentence):
        label = random.choice(sentence)
    return label
    
def get_rand_sentence(preprocess=True):
    f_time = time.time()
    print("Creating random sentences...")
    sentences = [" ".join(sent) for sent in brown.sents()[0:30000]]
    df = pd.DataFrame(sentences, columns=['context'])
    df = df.dropna(how = 'all') 
    df['is_drug'] = 0
    df['label'] = 'null'
    df = preprocess_exploded(df, preprocess)
    df['label'] = df.apply(rand_label, axis=1)
    df.drop(df[df['label'].str.len() < 3].index, inplace = True)
    df.drop(df[df['context'].str.len() < 4].index, inplace = True)
    df = df[~df['label'].str.contains(r'\d')]
    print("\tNumber of random sentences created: ", df.shape[0])
    print('\tTime taken to get random sentences:', time.time()-f_time)
    df['context'] = df.apply(replace_label, axis=1, args=['ludmyla'])
    df['context'] = df['context'].apply(lambda x: str(' '.join(x)))
    print(df)
    return df

def ungroup(df):
    print("Exploding columns")
    print("\tPrior to explode:", df.shape)
    df = df.explode('context')
    print("\tAfter explode: ", df.shape)
    return df
    
def create_df(resp_json,corpus,sorted_labels,val_t,val_f, context_size=80, max_context=5, group_labels=False):
    l_true = get_lines(val_t)
    l_false = get_lines(val_f)
    l_known = l_true + l_false
    l_unknown = [label for label in sorted_labels if label not in l_known]

    df = pd.DataFrame(sorted_labels, columns=['label'])

    df_known = df[df['label'].isin(l_known)]
    df_known.loc[:,('is_drug')] = df['label']
    df_known['is_drug'] = df_known['is_drug'].apply(lambda x: 1 if x in l_true else 0 if x in l_false else 2)
    print(df_known)
    text = get_text(corpus)
    df_known = add_context(df_known, text, context_size)
    if(max_context > 0 and group_labels):
        print('test')
        # df_known = limit_context(df_known, max_context)
    if not group_labels:
        df_known = ungroup(df_known)
    df_rand = get_rand_sentence()
    df_known = df_known.append(df_rand, ignore_index=True)
    df_known.to_csv(name_file('data_df_known.csv', '.csv'), index=False)
    print('\tKnown data build path: ', name_file('data_df_known.csv', '.csv'))

    df = pd.DataFrame(sorted_labels, columns=['label'])
    # df = df[df.label.isin(l_unknown)]
    # df['is_drug'] = 2
    # df = add_context(df, text, context_size)
    # if(max_context > 0):    
    #     df = limit_context(df, max_context)
    df.to_csv(name_file('data_df.csv', '.csv'), index=False)
    print('\tUnknown data build path: ', name_file('data_df.csv', '.csv'))

def limit_context_size(df_row, context_size, typec):
    context = df_row['context']
    if(typec == 'list'):
        context = ' '.join(context)
    if df_row['is_drug'] == 1:
        size = len(context)
        if size > context_size:
            cut = int((size - context_size) / 2)
            cut_l = context.rfind(' ', 0, cut)+1
            cut_r = context.rfind(' ', 0, size-cut)
            context = context[cut_l:cut_r]
            # context_1 = context[cut:size-cut]  
            # print('---TEST---')
            # print(size)
            # print(context)
            # print("cut: ", cut, "end: ", size-cut, "l: ", cut_l, "r: ", cut_r)
            # print(context_1)
            # print(context_2)
    if(typec == 'list'):
        context = context.split()
    return(context)

def create_df_unitex(uc_filepath,sorted_labels,val_t,val_f, context_size=80):
    print("Creating w2v database")
    l_true = get_lines(val_t)
    l_false = get_lines(val_f)
    l_known = l_true + l_false
    l_unknown = [label for label in sorted_labels if label not in l_known]

    df = pd.read_csv(uc_filepath)

    df_known = df[df['label'].isin(l_known)]
    df_known.loc[:,('is_drug')] = df['label']
    df_known['is_drug'] = df_known['is_drug'].apply(lambda x: 1 if x in l_true else 0 if x in l_false else 2)
    df_known['context'] = df['context'].apply(lambda x: eval(x))
    df_known = ungroup(df_known)
    df_known['context'] = df_known.apply(limit_context_size,axis=1,args=[context_size,'string'])
    df_rand = get_rand_sentence()
    df_known = df_known.append(df_rand, ignore_index=True)
    df_known = df_known[['label', 'is_drug','context']]
    print("\tCreated dataframe: ", df_known.shape)
    df_known.to_csv(name_file('data_df_known.csv', '.csv'), index=False)
    print('\tKnown data build path: ', name_file('data_df_known.csv', '.csv'))

    df = pd.DataFrame(columns=['label'])
    # df = df[df.label.isin(l_unknown)] 
    # df['is_drug'] = 2
    # df = add_context(df, text, context_size)
    # if(max_context > 0):    
    #     df = limit_context(df, max_context)
    df.to_csv(name_file('data_df.csv', '.csv'), index=False)
    print('\tUnknown data build path: ', name_file('data_df.csv', '.csv'))

#run case example: python3 ./misspelling.py extract [data file location] [type]
#ulyana use case: python3 ./rf_classifier.py build_data ./data/pitt-data_goodlines.txt drug drug_yes.txt drug_no.txt

#TODO: does not preprocess false correctly
def preprocess_to_sentence(train_topics, preprocess=True):
    train_sentences = []
    num_sentence = 0
    num_topics = 0
    for topic in train_topics:
        num_topics+=1
        topic_sentences = ast.literal_eval(topic)
        sentence_list = [sentence for sentence in topic_sentences]
        num_sentence += len(sentence_list)
        for sentence in sentence_list:
            if preprocess:
                sentence = re.sub("[^a-zA-Z-]"," ", sentence)
                # convert to lower case and split at whitespace
                words = sentence.split()
                # remove stop words (false by default)
                stops = set(stopwords.words("english"))
                words = [w for w in words if not w in stops]
                # sentence = " ".join(words)
                sentence = list(words)
            train_sentences.append(sentence)
    print("Number of train sentences: ", len(train_sentences))
    print("Number of topics: ", num_topics)
    
    return train_sentences

def main():

    args = sys.argv[1:]
    service = args[0]
    global group_types

    corpus = args[1]
    print("CORPUS: ", corpus)
    types = []

    if len(args) > 2:
        types = args[2:-3]
        group_types  = '_'.join(types)
        print("Types: ", types)
        print("Group Types: ", group_types)

    if service == 'build_data_norm':
        # python3 ./rf_classifier.py build_data_norm ./data/pitt-data_goodlines.txt drug drug_yes.txt drug_no.txt space
        resp = extract(corpus, types)
        resp_json = preprocess_json(resp.json())
        sorted_labels = create_sorted_labels(resp_json, False)
        val_t = args[-2]
        val_f = args[-1]
        create_df(resp_json, corpus, sorted_labels, val_t, val_f, context_size = 100, max_context=3, group_labels=False)
    elif service == 'build_data_unitex':
        # python3 ./rf_classifier.py build_data_unitex ./data/pitt-data_goodlines.txt drug drug_yes.txt drug_no.txt unitex_context_df_drug.csv
        resp = extract(corpus, types)
        resp_json = preprocess_json(resp.json())
        sorted_labels = create_sorted_labels(resp_json, False)
        val_t = args[-3]
        val_f = args[-2]
        unitex_context_filepath = args[-1]
        create_df_unitex(unitex_context_filepath, sorted_labels, val_t, val_f, context_size = 110)
    elif service == 'rf':
        # python3 ./rf_classifier.py rf ./data/pitt-data_goodlines.txt drug ./data_df_known_drug.csv ./data_df_drug.csv space
        df_known_file = args[-3]
        df_unknown_file = args[-2]
        rf_classifier(df_known_file, df_unknown_file)
    else:
        print('Incorrect service, use [build_data, rf] not:', service)

if __name__ == '__main__':
    main()
