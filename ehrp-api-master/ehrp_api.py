'''
Biomedical text processing API for EHR Phenotyping
'''

from __future__ import print_function
import os
import argparse
from flask import Flask, jsonify, request, abort
from flask_restful import Api, Resource, reqparse
from ehrp_utils import load_alphabets, free_alphabets, extract_concepts, get_json_from_file, RESOURCES_RELATIVE_PATH, dict_names_to_paths
from werkzeug.datastructures import FileStorage
from unitex import init_log_system
from unitex.config import UnitexConfig
import yaml

class Ehrs(Resource):
    '''Ehrs API for extracting Biomedical named entities and their respective concept IDs'''
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        # Define allowable arguments
        self.reqparse.add_argument('text', required=False, default=None, action='append',
                               location=['values', 'json'], help="text for extracting entities and concept ids")
        self.reqparse.add_argument('types', type=str, required=False, default=None, action='append',
                               location=['values', 'json'], help="type of concept to extract")
        self.reqparse.add_argument('file', type=FileStorage, required=False, default=None,
                               location='files', help="optional file to be parsed, at most one of 'text' or 'file' should have data")
        super(Ehrs, self).__init__()

    def post(self):
        '''POST method'''
        print("POST - Ehrs")     # for debugging

        # Get arguments
        args = self.reqparse.parse_args()
        text = args['text']
        types = args['types']
        file = args['file']

        # If user tries to use 'lookup' graph in extract operation or use 'master' specifically.
        if types and ( ('lookup' in types) or ('master' in types) ):
            print('''[ERROR] User tried to use \'lookup\' in extract operation
                  or specified master graph''')
            abort(422)

        # If both set or neither set
        if (text and file) or not(text or file):
            print('[ERROR] Either both text and file are being used, or neither are')
            abort(422)

        # If file is set, text isn't, and so we update text with the contents of file
        if file:
            # Assume file has one EHR per line
            text = [str(line) for line in file]

        # If no types specified, look for all types
        if types == None:
            concepts = extract_concepts(OPTIONS, ALL_GROUPINGS, ALL_DICTS_AND_ONTOLOGIES, text)
        # Otherwise use the types specified
        else:
            concepts = extract_concepts(OPTIONS, ALL_GROUPINGS, ALL_DICTS_AND_ONTOLOGIES, text, types)

        return jsonify(concepts)

class Terms(Resource):
    '''Terms API for finding the relevant concept ID'''
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('term', type=str, required=True, location='args',
                               help="text for finding concept id")
        super(Terms, self).__init__()

    def get(self):
        '''GET method'''
        print("GET - Terms")         # for debugging

        # Get argument
        args = self.reqparse.parse_args()

        # Make into a singleton list, as expected by 'extract_concepts'
        term = [args['term']]

        # Get results
        concepts = extract_concepts(OPTIONS, ALL_GROUPINGS, ALL_DICTS_AND_ONTOLOGIES, term, ['lookup'])

        return jsonify(concepts)

def main():
    '''Main method : parse arguments and start API'''
    global OPTIONS
    global ALL_GROUPINGS
    global ALL_DICTS_AND_ONTOLOGIES

    # Relative file path to user-defined json. Please update if project file-layout is changed.
    GRAMMAR_PARSING_GROUPS_PATH = os.path.join(RESOURCES_RELATIVE_PATH, 'GrammarParsingFunction.json')
    DICTS_AND_ONTOLOGIES_PATH = os.path.join(RESOURCES_RELATIVE_PATH, 'DictsAndOntologies.json')

    parser = argparse.ArgumentParser()
    # RESOURCE LOCATIONS
    parser.add_argument('--conf', type=str, default=os.path.join(RESOURCES_RELATIVE_PATH, 'unitex-med.yaml'),
                        help="Path to yaml file")

    # API SETTINGS
    parser.add_argument('--host', type=str, default='localhost',
                        help="Host name (default: localhost)")
    parser.add_argument('--port', type=int, default='8020', help="Port (default: 8020)")
    args = parser.parse_args()

    # Load resources
    config = None
    with open(args.conf, "r") as c_file:
        config = yaml.load(c_file)
    OPTIONS = UnitexConfig(config)
    init_log_system(OPTIONS["verbose"], OPTIONS["debug"], OPTIONS["log"])
    load_alphabets(OPTIONS)

    # Get all grammar and parsing function groupings
    ALL_GROUPINGS = get_json_from_file(GRAMMAR_PARSING_GROUPS_PATH)

    # Get all dictionary and ontology names
    ALL_DICTS_AND_ONTOLOGIES = get_json_from_file(DICTS_AND_ONTOLOGIES_PATH)
    ALL_DICTS_AND_ONTOLOGIES['dictionaries'] = dict_names_to_paths(ALL_DICTS_AND_ONTOLOGIES['dictionaries'])

    print("Starting app . . .")
    app = Flask(__name__)
    # Running DEBUG mode for flask. Makes JSON outputs more readable.
    ''''''
    app.config['DEBUG'] = True
    ''''''
    api = Api(app, prefix='/ehrp-api/v1')

    # Handle missing page 404 error
    @app.errorhandler(404)
    def page_not_found(error):
        '''Error message for page not found'''
        return "page not found : {}".format(error)

    # Handle internal server error
    @app.errorhandler(500)
    def raise_error(error):
        '''Error message for resource not found'''
        return error

    # Make extract and lookup pages available
    api.add_resource(Ehrs, '/ehrs')
    api.add_resource(Terms, '/terms')
    app.run(host=args.host, port=args.port)

    # Free resources
    free_alphabets(OPTIONS)


if __name__ == '__main__':
    main()
