'''
Utility file containing Unitex related utility methods
'''

from __future__ import print_function
from flask import abort
import os
import sys
import random
import string
import json
from pathlib import Path
from ConceptParser import ConceptParser
from unitex.io import ls, rm, exists, UnitexFile
from unitex.tools import UnitexConstants, normalize, tokenize
from unitex.resources import free_persistent_alphabet, load_persistent_alphabet

# Constants reflecting project file layout, please update if you change where files are stored.
RESOURCES_RELATIVE_PATH = 'resources'
GRAMMAR_RELATIVE_PATH = os.path.join(RESOURCES_RELATIVE_PATH, 'Grammars')
DICTIONARY_RELATIVE_PATH = os.path.join(RESOURCES_RELATIVE_PATH, 'Dictionaries')

# Called from ehrp_api.py
def load_alphabets(options):
    ''' Place alphabets in persistent space for re-use. '''
    options['resources']['alphabet'] = load_persistent_alphabet(options['resources']['alphabet'])
    options['resources']['alphabet-sorted'] = load_persistent_alphabet(options['resources']['alphabet-sorted'])

# Called from ehrp_api.py
def free_alphabets(options):
    ''' Remove alphabets from persistent space when api is shut down. '''
    free_persistent_alphabet(options['resources']['alphabet'])
    free_persistent_alphabet(options['resources']['alphabet-sorted'])

# Called from ehrp_api.py
def extract_concepts(options, all_groupings, text, concepts_to_get='ALL'):
    '''
    Extracts concepts from text.
    Returns dictionary of found concepts.
    '''

    print("Extracting concepts from text . . .")

    # Load chosen dict and grammar groupings from all_groupings
    chosen_groupings = get_concepts_from_groupings(all_groupings, concepts_to_get)

    # Create Unitex file to hold input text
    unifile = UnitexFile()
    text_file_name = random_filename()

    # Apply virtual file system prefix
    text_file_path = "%s%s" % (UnitexConstants.VFS_PREFIX, text_file_name + ".txt")

    # Write input text to unitex text file
    unifile.open(text_file_path, mode='w')
    unifile.write(str(text))
    unifile.close()

    # Get directory file path
    directory, _ = os.path.split(text_file_path)

    # Set snt file path (snt files are .txt files that have been processed by Unitex)
    snt = os.path.join(directory, "%s.snt" % text_file_name)
    snt = "%s%s" % (UnitexConstants.VFS_PREFIX, snt)
    # Place snt file into snt directory
    dirc = os.path.join(directory, "%s_snt" % text_file_name)

    # Get Alphabets
    alphabet_unsorted = options["resources"]["alphabet"]
    alphabet_sorted = options["resources"]["alphabet-sorted"]

    # Normalize the text
    normalize_text(text_file_path, options["tools"]["normalize"])

    # Tokenize the text
    tokenize_text(snt, alphabet_unsorted, options["tools"]["tokenize"])

    # Get concepts that match grammars
    concepts = get_concepts_for_grammars(dirc, options, snt, alphabet_unsorted, alphabet_sorted, chosen_groupings)

    # Clean the Unitex files
    print("Cleaning up files from " + dirc)
    for v_file in ls("%s%s" % (UnitexConstants.VFS_PREFIX, dirc)):
        rm(v_file)
    rm(snt)
    rm(text_file_path)

    return concepts

# function: get_concepts_for_grammars; Returns a list of dictionary objects of parsed concepts from text
# options: yaml object with preset options for different unitex functions
# snt: the file path to the pre-processed text
# alphabet_unsorted: file path to alphabet unitex should use, unsorted
# alphabet_sorted: file path to alphabet unitex should use, sorted
# groupings: list of dictionary objects holding info from GRAMMAR_DICTIONARY_PARSING_GROUPS_PATH
def get_concepts_for_grammars(directory, options, snt, alphabet_unsorted, alphabet_sorted, concepts):
    list_of_concepts = []

    # Set arguments that don't change across grammar/dictionary usage
    concept_parser = ConceptParser(
        directory = directory,
        options = options,
        text = snt,
        alphabet_unsorted = alphabet_unsorted,
        alphabet_sorted = alphabet_sorted
    )

    # Set concept_parser grammar, dictionaries, and parsing_functions to those in GrammarDictionaryParsingFunction.py
    for grammar_dictionary_parser in concepts:
        grammar_path = os.path.join(GRAMMAR_RELATIVE_PATH, grammar_dictionary_parser['grammar'])
        dictionary_paths = [ os.path.join(DICTIONARY_RELATIVE_PATH, dictionary) for dictionary in grammar_dictionary_parser['dictionaries'] ]

        concept_parser.grammar = grammar_path
        concept_parser.dictionaries = dictionary_paths
        concept_parser.ontology_names = grammar_dictionary_parser['ontologies']
        concept_parser.parsing_function = grammar_dictionary_parser['parsing_function']

        # Make use of ConceptParser member variables that might not be set during object construction
        # Maps parsing_function string to function reference
        concept_parser.setup()

        # Process snt using concept_parser.grammar, concept_parser.dictionaries, and concept_parser.parsing_function
        concepts = concept_parser.parse()

        # Append only if at least one concept found.
        if len(concepts['instances']):
            list_of_concepts.append(concepts)

    return list_of_concepts

def get_groupings_from_file(file_path):
    ''' Loads the user-chosen groupings of grammars, dictionaries, and parsing functions as a dictionary '''
    with open(file_path) as file:
        groupings = json.load(file)
    return groupings

def get_concepts_from_groupings(all_groupings, concepts_to_get):
    ''' Returns list of concepts to get as specified by user '''
    concepts = []

    # If all concepts are desired, just return all groupings
    if concepts_to_get == 'ALL':
        return all_groupings

    for concept in concepts_to_get:
        # Match desired concepts with associated grouping
        for grouping in all_groupings:
            # Accepted concept types are specified in GrammarDictionaryParsingFunction.json
            if grouping['grammar'] == concept+'.fst2':
                concepts.append(grouping)
                break
        # If concept not found, must be incorrect
        else:
            incorrect_concept_type(concept)
    return concepts

def random_filename(size=8, chars=string.ascii_uppercase + string.digits):
    '''Returns a random string'''
    return ''.join(random.choice(chars) for _ in range(size))

def normalize_text(text, kwargs):
    ''' Creates .snt file of the normalized text '''
    # normalize returns True on succeess, False on failure
    normalization_succeeded = normalize(text, **kwargs)

    if normalization_succeeded is False:
        sys.stderr.write("[ERROR] Text normalization failed!\n")

def tokenize_text(snt_file_path, alphabet, kwargs):
    ''' Creates file of tokens '''
    tokenization_succeeded = tokenize(snt_file_path, alphabet, **kwargs)

    if tokenization_succeeded is False:
        sys.stderr.write("[ERROR] Text tokenization failed!\n")
        sys.exit(1)

# TODO: Raise exception that can be handled, to return a more descriptive error
def incorrect_concept_type(incorrect_type):
    # Unprocessable entity error
    abort(422)
