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

def get_matches(resp_json, corpus, sorted_labels, ratio=0.8, context_size=100, max_context=-1, context=True, group_labels=True):
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

def to_word_list(sentence, preprocess):
    if preprocess:
        sentence = re.sub("[^a-zA-Z-]"," ", str(sentence))
        # convert to lower case and split at whitespace
        words = sentence.split()
        # remove stop words (false by default)
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        # sentence = " ".join(words)
        sentence = list(words)
    else:
        words = sentence.split()
        sentence = list(words)

    return sentence
    
def preprocess_exploded(df, preprocess=True):
    df['context'] = df['context'].apply(to_word_list, args=[preprocess])
    return df

def rf_classifier(df_known_file, df_unknown_file, exploded=True, preprocess=True):
    # Load in data
    df = pd.read_csv(df_known_file)
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    # df['context'] = df['context'].apply(preprocess_to_wordlist)
    if exploded:
        df = preprocess_exploded(df, preprocess)
        print("Whole Dataframe size:\n", df.shape)
        print("Training Size possible: ", int(df.shape[0] * 0.75))
        df_train = df.sample(n=int(df.shape[0]* 0.75))
        print("Training size: ", df_train.shape)
        df_test = df[~df.isin(df_train)].dropna()
        print("Testing size: ", df_test.shape)
        train_sentences = df_train['context'].to_list()
        print("number train sentences: ", len(train_sentences))
    else:
        train_topics = df['context'].to_list()
        train_sentences = preprocess_to_sentence(train_topics)
   
    model_name = 'drug_train_model'
    # Set values for various word2vec parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 3   # Minimum word count                        
    num_workers = 3       # Number of threads to run in parallel
    context = 200          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    if not os.path.exists(model_name): 
        # Initialize and train the model (this will take some time)
        model = word2vec.Word2Vec(train_sentences, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling)

        # If you don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)
        model.save(model_name)
    else:
        model = Word2Vec.load(model_name)

    print("Word2Vec model heuristics: ", model)

    # calculate average feature vectors for training and test sets
    clean_train_reviews = []
    for review in df_train['context']:
        clean_train_reviews.append(review)
    trainDataVecs = get_avg_feature_vecs(clean_train_reviews, model, num_features)

    nan_indices = list({x for x,y in np.argwhere(np.isnan(trainDataVecs))})
    if len(nan_indices) > 0:
        print('Removing {:d} instances from test set.'.format(len(nan_indices)))
        trainDataVecs = np.delete(trainDataVecs, nan_indices, axis=0)
        df_train.drop(df_train.iloc[nan_indices, :].index, axis=0, inplace=True)
        assert trainDataVecs.shape[0] == len(df_train)

    clean_test_reviews = []
    for review in df_test['context']:
        clean_test_reviews.append(review)
    testDataVecs = get_avg_feature_vecs(clean_test_reviews, model, num_features)

    nan_indices = list({x for x,y in np.argwhere(np.isnan(testDataVecs))})
    if len(nan_indices) > 0:
        print('Removing {:d} instances from test set.'.format(len(nan_indices)))
        testDataVecs = np.delete(testDataVecs, nan_indices, axis=0)
        df_test.drop(df_test.iloc[nan_indices, :].index, axis=0, inplace=True)
        assert testDataVecs.shape[0] == len(df_test)

    forest = RandomForestClassifier(n_estimators = 100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, df_train['is_drug'])

    print("Predicting labels for test data..")
    result = forest.predict(testDataVecs)

    print(classification_report(df_test['is_drug'], result))
    probs = forest.predict_proba(testDataVecs)[:, 1]

    fpr, tpr, _ = roc_curve(df_test['is_drug'], probs)
    auc = roc_auc_score(df_test['is_drug'], probs)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC {:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()




def make_feature_vec(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
    nwords = 0.
    index2word_set = set(model.wv.index2word)  # words known to the model
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec,model[word])

    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    """
    Calculate average feature vectors for all reviews
    """
    counter = 0
    review_feature_vecs = np.zeros((len(reviews),num_features), dtype='float32')  # pre-initialize (for speed)
    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter+=1
    return review_feature_vecs

    # print(model.wv.vocab)

    # Extract the labels
    # labels = np.array(df.pop('is_drug'))
    # print(labels)

    # # 30% examples in test data
    # train, test, train_labels, test_labels = train_test_split(df,
    #                                                         labels, 
    #                                                         stratify = labels,
    #                                                         test_size = 0.3, 
    #                                                         random_state = RSEED)

    # # Imputation of missing values
    # # train = train.fillna(train.mean())
    # # test = test.fillna(test.mean())

    # # Features for feature importances
    # features = list(train.columns)
    # print(features)
    # # Create the model with 100 trees
    # model = RandomForestClassifier(n_estimators=100, 
    #                                 random_state=RSEED, 
    #                                 max_features = 'sqrt',
    #                                 n_jobs=-1, verbose = 1)

    # # Fit on training data
    # model.fit(train, train_labels)
    # print("model fit")

    # n_nodes = []
    # max_depths = []

    # # Stats about the trees in random forest
    # for ind_tree in model.estimators_:
    #     n_nodes.append(ind_tree.tree_.node_count)
    #     max_depths.append(ind_tree.tree_.max_depth)

    # print(f'Average number of nodes {int(np.mean(n_nodes))}')
    # print(f'Average maximum depth {int(np.mean(max_depths))}')

    # # Training predictions (to demonstrate overfitting)
    # train_rf_predictions = model.predict(train)
    # train_rf_probs = model.predict_proba(train)[:, 1]

    # # Testing predictions (to determine performance)
    # rf_predictions = model.predict(test)
    # rf_probs = model.predict_proba(test)[:, 1]

#    from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
    # import matplotlib.pyplot as plt

    # # Plot formatting
    # plt.style.use('fivethirtyeight')
    # plt.rcParams['font.size'] = 18

def ungroup(df):
    print("Exploding columns")
    print("\tPrior to explode:", df.shape)
    df = df.explode('context')
    print("\tAfter explode: ", df.shape)
    return df

def create_df(resp_json,corpus,sorted_labels,val_t,val_f, context_size=200, max_context=5, group_labels=False):
    l_true = get_lines(val_t)
    l_false = get_lines(val_f)
    l_known = l_true + l_false
    l_unknown = [label for label in sorted_labels if label not in l_known]

    df = pd.DataFrame(sorted_labels, columns=['label'])

    df_known = df[df['label'].isin(l_known)]
    df_known.loc[:,('is_drug')] = df['label']
    df_known['is_drug'] = df_known['is_drug'].apply(lambda x: 1 if x in l_true else 0 if x in l_false else 2)
    text = get_text(corpus)
    df_known = add_context(df_known, text, context_size)
    if(max_context > 0 and group_labels):
        print('test')
        # df_known = limit_context(df_known, max_context)
    if not group_labels:
        df_known = ungroup(df_known)
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
        types = args[2:-2]
        group_types  = '_'.join(types)
        print("Types: ", types)
        print("Group Types: ", group_types)

    if service == 'build_data':
        # python3 ./rf_classifier.py build_data ./data/pitt-data_goodlines.txt drug drug_yes.txt drug_no.txt
        resp = extract(corpus, types)
        resp_json = preprocess_json(resp.json())
        sorted_labels = create_sorted_labels(resp_json, False)
        val_t = args[-2]
        val_f = args[-1]
        create_df(resp_json, corpus, sorted_labels, val_t, val_f, max_context=3, group_labels=False)
    elif service == 'rf':
        # python3 ./rf_classifier.py rf ./data/pitt-data_goodlines.txt drug ./data_df_known_drug.csv ./data_df_drug.csv
        df_known_file = args[-2]
        df_unknown_file = args[-1]
        rf_classifier(df_known_file, df_unknown_file)
    else:
        print('Incorrect service, use [build_data, rf] not:', service)
    
    remove_file(name_file('no_key_word.txt'))

if __name__ == '__main__':
    main()
