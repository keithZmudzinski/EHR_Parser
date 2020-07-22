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
from unitex.io import ls, rm, exists, UnitexFile, mv
from unitex.tools import UnitexConstants, normalize, tokenize, dico
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
def extract_concepts(options, all_groupings, dicts_and_ontos, text, concepts_to_get='ALL'):
    '''
    Extracts concepts from text.
    Returns dictionary of found concepts.
    '''

    print("Extracting concepts from text . . .")

    # Get Alphabets
    alphabet_unsorted = options["resources"]["alphabet"]
    alphabet_sorted = options["resources"]["alphabet-sorted"]

    # Put all texts together for preprocessing
    combined_text = '\n\n'.join(text)

    # Create folder in virtual file system
    combined_text_filename = to_VFS(random_filename())
    combined_text_path = combined_text_filename + ".txt"

    # Save combined text in file in VFS
    unitex_file = UnitexFile()
    unitex_file.open(combined_text_path, mode='w')
    unitex_file.write(combined_text)
    unitex_file.close()

    # Normalize the combined text
    normalize_text(combined_text_path, options["tools"]["normalize"])

    # Get file path of normalized text
    combined_processed_text_path = combined_text_filename + ".snt"

    # Tokenize the text (alters combined_processed_text in place)
    tokenize_text(combined_processed_text_path, alphabet_unsorted, options["tools"]["tokenize"])

    # Apply dictionaries
    apply_dictionaries(dicts_and_ontos['dictionaries'], combined_processed_text_path, alphabet_unsorted, options)

    # Create a text file in the VFS for each health record
    health_record_paths = []
    for record_number, health_record in enumerate(text):
        # Create file name that will hold content
        health_record_file_name = to_VFS("text_%d" % record_number)
        # Place file into VFS
        health_record_path = health_record_file_name + ".txt"

        # Save content in Unitex file
        unitex_file.open(health_record_path, mode='w')
        unitex_file.write(health_record)
        unitex_file.close()

        # Pre-process this file to allow for later processing
        normalize_text(health_record_path, options["tools"]["normalize"])
        health_record_processed_text_path = health_record_file_name + ".snt"
        # Save processed file path for later processing
        health_record_paths.append(health_record_processed_text_path)
        tokenize_text(health_record_processed_text_path, alphabet_unsorted, options["tools"]["tokenize"])

    # Load chosen dict and grammar groupings from all_groupings
    chosen_groupings = get_concepts_from_groupings(all_groupings, concepts_to_get)

    # Get concepts that match grammars
    concepts_per_ehrp = []
    prev_dic_files_locations = {
        "dlf": os.path.join(combined_text_filename + "_snt", "dlf"),
        "dlc": os.path.join(combined_text_filename + "_snt", "dlc")
    }
    for record_number, health_record_path in enumerate(health_record_paths):
        # Get the folder in which all files for this health record are stored
        health_record_folder = to_VFS("text_%d_snt" % record_number)
        # Need to place dictionary files into this folder
        new_dlf_path = os.path.join(health_record_folder, "dlf")
        new_dlc_path = os.path.join(health_record_folder, "dlc")

        # Place the dictionary files into this folder
        mv(prev_dic_files_locations["dlf"], new_dlf_path)
        mv(prev_dic_files_locations["dlc"], new_dlc_path)

        # Update location of dictionay files
        prev_dic_files_locations["dlf"] = new_dlf_path
        prev_dic_files_locations["dlc"] = new_dlc_path

        # Apply graphs to text and get found concepts
        concepts = get_concepts_for_grammars(health_record_folder, options, health_record_path,
                                            alphabet_unsorted, alphabet_sorted,
                                            chosen_groupings, dicts_and_ontos['ontologies']
                                            )
        concepts_per_ehrp.append(concepts)

    # Clean the Unitex files
    print("Cleaning up files")
    for v_file in ls("%s" % UnitexConstants.VFS_PREFIX):
        rm(v_file)

    return concepts_per_ehrp

# function: get_concepts_for_grammars; Returns a list of dictionary objects of parsed concepts from text
# directory: virtual file system directory
# options: yaml object with preset options for different unitex functions
# snt: the file path to the pre-processed text
# alphabet_unsorted: file path to alphabet unitex should use, unsorted
# alphabet_sorted: file path to alphabet unitex should use, sorted
# chosen_groupings: The groupings from GrammarParsingFunction.json that will be applied to the input text
# ontologies: the names of the ontologies being used, allows dictionary file names to differ from the ontology they are using
def get_concepts_for_grammars(directory, options, snt, alphabet_unsorted, alphabet_sorted, chosen_groupings, ontologies):
    list_of_concepts = []

    # Set arguments that don't change across grammar/dictionary usage
    concept_parser = ConceptParser(
        directory = directory,
        options = options,
        text = snt,
        alphabet_unsorted = alphabet_unsorted,
        alphabet_sorted = alphabet_sorted,
        ontology_names = ontologies
    )

    # Set concept_parser grammar, dictionaries, and parsing_functions to those in GrammarDictionaryParsingFunction.py
    for grammar_dictionary_parser in chosen_groupings:
        grammar_path = os.path.join(GRAMMAR_RELATIVE_PATH, grammar_dictionary_parser['grammar'])

        concept_parser.grammar = grammar_path
        concept_parser.parsing_function = grammar_dictionary_parser['parsing_function']

        # Make use of ConceptParser member variables that might not be set during object construction
        # Maps parsing_function string to function reference
        concept_parser.setup()

        # Process snt using concept_parser.grammar, concept_parser.dictionaries, and concept_parser.parsing_function
        concepts = concept_parser.parse()

        try:
            # Append only if at least one concept found.
            if len(concepts['instances']):
                list_of_concepts.append(concepts)
        # This happens if we are parsing the master graph
        except TypeError:
            list_of_concepts.extend(concepts)

    return list_of_concepts

def apply_dictionaries(dictionaries, text, alphabet_unsorted, options):
        ''' Creates .dlf and .dlc files holding words in both dictionaries and text. '''
        if dictionaries is not None:
            dictionaries_applied_succesfully = dico(dictionaries, text, alphabet_unsorted, **options['tools']['dico'])

            if dictionaries_applied_succesfully is False:
                sys.stderr.write("[ERROR] Dictionaries application failed!\n")
                sys.exit(1)
        else:
            sys.stderr.write("[ERROR] No dictionaries specified.\n")

def get_json_from_file(file_path):
    ''' Loads the user-chosen groupings of grammars, dictionaries, and parsing functions as a dictionary '''
    with open(file_path) as file:
        groupings = json.load(file)
    return groupings

def get_concepts_from_groupings(all_groupings, concepts_to_get):
    ''' Returns list of concepts to get as specified by user '''
    concepts = []

    # If all concepts desired, use master graph and any sub-graphs not included in master graph.
    # This is faster than making many individual queries.
    if concepts_to_get == 'ALL':
        for grouping in all_groupings:
            is_sub_graph = grouping.get('sub_graph', False)
            # Only accept sub_graphs with 'True' as value,
            #  in case programmer places 'False' as value.
            if is_sub_graph == 'True':
                concepts.append(grouping)
            elif grouping['grammar'] == 'master.fst2':
                concepts.append(grouping)
        return concepts

    # If specific concepts are desired
    for concept in concepts_to_get:
        # Match desired concepts with associated grouping
        for grouping in all_groupings:
            # Accepted concept types are specified in GrammarDictionaryParsingFunction.json
            if grouping['grammar'] == concept + '.fst2':
                concepts.append(grouping)
                break
        # If concept not found, must be incorrectly specified, so abort
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

def dict_names_to_paths(dict_names):
    ''' Changes dictionary names to dictionary paths '''
    return [os.path.join(DICTIONARY_RELATIVE_PATH, name) for name in dict_names]

def to_VFS(input):
    return "%s%s" % (UnitexConstants.VFS_PREFIX, input)

# # FOR BATCH PROCESSING
#     # Place already created dictionaries into vfs
#         if large_query_detected:
#             new_dlf_path = "%s%s" % (UnitexConstants.VFS_PREFIX, os.path.join(self.directory, "dlf"))
#             new_dlc_path = "%s%s" % (UnitexConstants.VFS_PREFIX, os.path.join(self.directory, "dlc"))
#             start = time()
#             cp('resources/test_dictionaries/dlf', new_dlf_path)
#             print('Time taken to cp dlf:', time()-start)
#             start = time()
#             cp('resources/test_dictionaries/dlc', new_dlc_path)
#             print('Time taken to cp dlc:', time()-start)
#
#             simple_words = new_dlf_path
#             compound_words = new_dlf_path
