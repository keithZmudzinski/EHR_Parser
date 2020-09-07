import numpy as np
import random
import time
import requests
import os
import sys
import re

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
    num_documents = 914
    num_compound_words_to_make = 1

    # Where to save the model
    output_path = os.path.join(output_folder, model_name)

    # Get ehrs and separate them
    train_ehrs, test_ehrs = get_ehrs(pitt_path, num_documents)

    # Get tagger results
    train_results = make_query(train_ehrs, type)
    test_results = make_query(test_ehrs, type)

    # Create compound words from monosemous terms
    train_compound_words = make_compound_words(train_results, num_compound_words_to_make, dic_path)
    test_compound_words = make_compound_words(test_results, num_compound_words_to_make, dic_path, train_compound_words)

    # Get instances of compound words
    train_data = extract_words(train_results, train_compound_words)
    test_data = extract_words(test_results, test_compound_words)
    print('TRAINING DATA')
    print(train_data,'\n\n')
    print('TEST DATA')
    print(test_data, '\n\n')

    # Modify the data somehow if necessary
    train_data = feature_engineer(train_data)
    test_data = feature_engineer(test_data)

    os.environ['model_location'] = 'this is keith'
    # Create new model
    wsd = WSD()

    # CREATE MODEL IN SOME FASHION
    # wsd.model = SOMETHING

    # Save the dictionary before modifying it
    save_dic(dic_path, 'saved_dic')

    # Train and test data don't overlap, so okay to add both at same time
    add_to_dic(train_compound_words)
    add_to_dic(test_compound_words)

    # Replace monosemous terms with compound terms
    updated_train_ehrs = update_ehrs(train_ehrs, train_compound_words)
    updated_test_ehrs = update_ehrs(test_ehrs, test_compound_words)

    # Get results after using model
    train_evaluation_results = make_query(updated_train_ehrs, type)
    test_evaluation_results = make_query(updated_test_ehrs, type)

    # Just get parts we care about
    train_evaluation_results = parse_results(train_evaluation_results, train_compound_words)
    test_evaluation_results = parse_results(test_evaluation_results, test_compound_words)

    # Combine with known info
    train_x_and_y = combine_x_and_y(train_evaluation_results, train_data)
    test_x_and_y = combine_x_and_y(test_evaluation_results, test_data)

    # Get evaluation of model
    evaluate(train_x_and_y)
    evaluate(test_x_and_y)

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

def make_compound_words(results, num_to_make, dic_path, exclude=[]):
    words = set()
    with open(dic_path, 'r') as dic_file:
        dic_contents = dic_file.readlines()

    for ehr in results:
        instances = get_instances(ehr)
        # Make set of term/cuis found in text
        for instance in instances:
            words.add((instance['term'], instance['cui']))

    compound_words = []
    for _ in range(num_to_make):
        try:
            word1 = get_monoseme(words, dic_contents)
            word2 = get_monoseme(words, dic_contents)

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

            # Cols look like: Compound word, context, correct word, correct cui
            instance_to_add = np.array([compound_word, context, correct_word, correct_cui])
            data = np.vstack((data, instance_to_add))

    # Delete header row
    data = np.delete(data, 0, 0)
    return data

def get_monoseme(word_set, dic_contents):
    word = word_set.pop()

    while(is_homonym(word[0], dic_contents)):
        word = word_set.pop()

    word = {
        'term': word[0],
        'cui': word[1]
    }
    return word

def is_homonym(word, dic_contents):
    for line in dic_contents:
        term, info = unescaped_split('\\,', line)
        lemma, _ = unescaped_split('\\.', info)

        if term == word:
            return lemma == 'HOMONYM'

def unescaped_split(delimiter, line):
    # Only split on unescaped versions of delimiter in line
    return re.split(r'(?<!\\){}'.format(delimiter), line)

def evaluate(model, data):
    # generate bunha info and scores
    pass

def feature_engineer(data):
    ''' Modify features in some way '''
    pass

if __name__ == '__main__':
    main()
