import numpy as np
import random
import time
import requests
import os
import sys
import re

# Model imports
from sklearn.neighbors import KNeighborsClassifier

def add_to_path(path):
    sys.path.append(path)

add_to_path(os.path.join(os.path.dirname(sys.path[0]), 'ehrp-api-master'))
from WordSenseDisambiguation import WSD

add_to_path(os.path.join(os.path.dirname(sys.path[0]),'FindProblemEntry'))
from APIProcess import APIProcess

def main():
    output_folder = '../../WSD_Data'
    pitt_path = '../../English/Corpus/pitt-delimited.txt'

    # MODIFY THESE WHEN CHANGING WHAT MODEL IS LEARNING
    type = 'drug'
    dic_path = 'resources/Dictionaries/' + 'drug.dic'
    model_name = 'test_model'

    # Must be at least 10 EHRS
    num_documents = 200
    num_compound_words_to_make = 1

    # Where to save the model
    output_path = os.path.join(output_folder, model_name)

    # Get ehrs and separate them
    train_ehrs, test_ehrs = get_ehrs(pitt_path, num_documents)

    # Get tagger results
    train_results = make_query(train_ehrs, type)
    test_results = make_query(test_ehrs, type)

    # Create compound words from monosemous terms
    compound_words = make_compound_words(train_results, test_results, num_compound_words_to_make, dic_path)

    # Get instances of compound words
    train_data = extract_words(train_results, compound_words)
    test_data = extract_words(test_results, compound_words)

    # Modify the data somehow if necessary
    train_X, train_Y = feature_engineer(train_data)
    test_X, test_Y = feature_engineer(test_data)

    print('TRAIN DATA')
    print('TRAIN X')
    for thing in train_X:
        print(thing)
    print('TRAIN Y')
    for thing in train_Y:
        print(thing)

    print('TEST DATA')
    print('TEST X')
    for thing in test_X:
        print(thing)
    print('TEST Y')
    for thing in test_Y:
        print(thing)

    os.environ['model_location'] = 'this is keith'
    # Create new model
    wsd = WSD('NEW_MODEL')

    # CREATE MODEL IN SOME FASHION
    # wsd.model = SOMETHING

    # Get results on training and testing data
    train_results = apply_model(train_data)
    test_results = apply_model(test_data)

    # Get evaluation of model
    evaluate(train_results)
    evaluate(test_results)

    # Save model for re-use
    wsd.save()

def get_ehrs(path, num_documents):
    pitt_file = open(path, 'r')
    with open(path, 'r') as pitt_file:
        pitt_ehrs = [ehr.split('|')[7] for ehr in pitt_file]

    # Bound num documents between 0 and the max number of docs in pitt
    num_documents = min(len(pitt_ehrs), max(0, num_documents))

    # Get desired number of documents to test
    ehrs = random.sample(pitt_ehrs, num_documents)

    # Split data into training / testing
    data_split = int(len(ehrs) * .8)
    train_ehrs = ehrs[:data_split]
    test_ehrs = ehrs[data_split:]

    return train_ehrs, test_ehrs

def make_query(ehrs, type):
    # Constant
    extract_url = 'http://localhost:8020/ehrp-api/v1/ehrs'

    # Start the api
    api = APIProcess(0)
    api.start()
    # Wait for it to finish initializing
    time.sleep(2.6)

    # Setup arguments
    args = {
        'text': ehrs,
        'types': type
    }

    # Make the query
    response = requests.post(extract_url, data=args)

    # Kill the api
    api.terminate()

    return response.json()

def get_instances(ehr):
    ''' Break ehr results down into a list of instances '''
    # Should only be one type
    types = [type for type in ehr]
    lists_of_instances = [type['instances'] for type in types]

    instances = [instance for list_of_instances in lists_of_instances for instance in list_of_instances]
    return instances

def make_compound_words(train_results, test_results, num_to_make, dic_path):
    train_words = set()
    test_words = set()

    with open(dic_path, 'r') as dic_file:
        dic_contents = dic_file.readlines()

    for ehr in train_results:
        instances = get_instances(ehr)
        # Make set of term/cuis found in text
        for instance in instances:
            train_words.add((instance['term'], instance['cui']))

    for ehr in test_results:
        instances = get_instances(ehr)
        # Make set of term/cuis found in text
        for instance in instances:
            test_words.add((instance['term'], instance['cui']))

    compound_words = []
    for _ in range(num_to_make):
        # Get two monosemous words, each in both train and testing set
        try:
            word1 = get_monoseme(train_words, test_words, dic_contents)
            word2 = get_monoseme(train_words, test_words, dic_contents)

        # words became empty, so no more possible compounds words to be made
        except KeyError:
            break

        compound_term = word1['term'] + '-' + word2['term']
        compound_word = {
            'compound_term': compound_term,
            'term1': word1['term'],
            'term2': word2['term'],
            'cui1': word1['cui'],
            'cui2': word2['cui']
        }
        compound_words.append(compound_word)

    return compound_words

def find_compound_word(atom, compound_words):
    for compound_word in compound_words:
        # If the atom is part of the compound word
        if atom == compound_word['term1'] or atom == compound_word['term2']:
            return compound_word
    return None

def extract_words(results, compound_words):
    ''' Returns numpy table of compound words in corpus '''
    # Get list of instances from all ehrs in results
    instances = []
    for ehr in results:
        instances.extend(get_instances(ehr))

    # Look at each instance, if part of a compound word, add it to data
    data = np.array(['compound_word', 'context', 'correct_word', 'correct_cui'])
    for instance in instances:
        found_compound = find_compound_word(instance['term'], compound_words)
        if found_compound:
            compound_word = found_compound['compound_term']
            context = instance['context']
            correct_word = instance['term']
            correct_cui = instance['cui']

            # Replace the term with compound term in the context
            context = context.replace(correct_word, compound_word)
            
            # Cols look like: Compound word, context, correct word, correct cui
            instance_to_add = np.array([compound_word, context, correct_word, correct_cui])
            data = np.vstack((data, instance_to_add))

    # Delete header row
    data = np.delete(data, 0, 0)
    return data

def get_monoseme(train_word_set, test_word_set, dic_contents):
    word_is_monoseme = False

    # Loop until our word is a monoseme and in both word sets
    while(not(word_is_monoseme)):
        word = get_word_in_both(train_word_set, test_word_set)
        word_is_monoseme = True if is_monoseme(word[0], dic_contents) else False

    word = {
        'term': word[0],
        'cui': word[1]
    }

    return word

def get_word_in_both(train_word_set, test_word_set):
    # Get a word that is in training set
    word = train_word_set.pop()

    # Loop until we find a word that is in both training and testing set
    while(not(word in test_word_set)):
        word = train_word_set.pop()

    return word

def is_monoseme(word, dic_contents):
    for line in dic_contents:
        term, info = unescaped_split('\\,', line)
        lemma, _ = unescaped_split('\\.', info)

        # Always finds term, since term comes from the dic
        if term == word:
            return lemma != 'HOMONYM'

def unescaped_split(delimiter, line):
    # Only split on unescaped versions of delimiter in line
    return re.split(r'(?<!\\){}'.format(delimiter), line)

def evaluate(model, data):
    # generate bunha info and scores
    pass

def feature_engineer(data):
    ''' Modify features in some way '''

    # Split data into inputs and expected outputs
    return data[:, :2], data[:, 2]

if __name__ == '__main__':
    main()
