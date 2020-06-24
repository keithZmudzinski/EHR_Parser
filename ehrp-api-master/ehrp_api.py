'''
Biomedical text processing API for EHR Phenotyping
'''

from __future__ import print_function
import os
import argparse
from flask import Flask, jsonify, request, abort
from flask_restful import Api, Resource, reqparse
from ehrp_utils import load_alphabets, free_alphabets, extract_concepts, get_groupings_from_file
from werkzeug.datastructures import FileStorage
from unitex import init_log_system
from unitex.config import UnitexConfig
import yaml

class Extract(Resource):
    '''Extract API for extracting Biomedical named entities and their respective concept IDs'''
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        # Define allowable arguments
        self.reqparse.add_argument('text', required=False, default=None,
                               location='values', help="text for extracting entities and concept ids")
        self.reqparse.add_argument('types', type=str, required=False, default=None, action='append',
                               location='values', help="type of concept to extract")
        self.reqparse.add_argument('file', type=FileStorage, required=False, default=None,
                               location='files', help="optional file to be parsed, at most one of 'text' or 'file' should have data")
        super(Extract, self).__init__()

    def post(self):
        '''POST method'''
        print("POST - Extract")     # for debugging

        # Get arguments
        args = self.reqparse.parse_args()
        text = args['text']
        types = args['types']
        file = args['file']

        # If user tries to use 'lookup' graph in extract operation
        if 'lookup' in types:
            abort(422)

        # If both set or neither set
        if text and file or not(text or file):
            abort(422)

        # If file is set, text isn't, and so we update text with the contents of file
        if file:
            text = file.read()

        # If no types specified, look for all types
        if types == None:
            concepts = extract_concepts(OPTIONS, ALL_GROUPINGS, text)
        # Otherwise use the types specified
        else:
            concepts = extract_concepts(OPTIONS, ALL_GROUPINGS, text, types)

        return jsonify(concepts)

class Lookup(Resource):
    '''Lookup API for finding the relevant concept ID'''
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('text', type=str, required=True, location='args',
                               help="text for finding concept id")
        super(Lookup, self).__init__()

    def get(self):
        '''GET method'''
        print("GET - Lookup")         # for debugging

        # Get argument
        args = self.reqparse.parse_args()
        text = args['text']

        # Get results
        concepts = extract_concepts(OPTIONS, ALL_GROUPINGS, text, ['lookup'])

        return jsonify(concepts)

def main():
    '''Main method : parse arguments and start API'''
    global OPTIONS
    global ALL_GROUPINGS

    # Relative file path to user-defined json. Pleas update if project file-layout is changed.
    GRAMMAR_DICTIONARY_PARSING_GROUPS_PATH = os.path.join('resources', 'GrammarDictionaryParsingFunction.json')

    parser = argparse.ArgumentParser()
    # RESOURCE LOCATIONS
    parser.add_argument('--conf', type=str, default='resources/unitex-med.yaml',
                        help="Path to yaml file")

    # API SETTINGS
    parser.add_argument('--host', type=str, default='localhost',
                        help="Host name (default: localhost)")
    parser.add_argument('--port', type=int, default='8020', help="Port (default: 8020)")
    parser.add_argument('--path', type=str, default='/ehrp', help="Path (default: /ehrp)")
    args = parser.parse_args()

    # Load resources
    config = None
    with open(args.conf, "r") as c_file:
        config = yaml.load(c_file)
    OPTIONS = UnitexConfig(config)
    init_log_system(OPTIONS["verbose"], OPTIONS["debug"], OPTIONS["log"])
    load_alphabets(OPTIONS)

    # Get all grammar, dictionary, parsing function groupings
    ALL_GROUPINGS = get_groupings_from_file(GRAMMAR_DICTIONARY_PARSING_GROUPS_PATH)

    print("Starting app . . .")
    app = Flask(__name__)
    api = Api(app)

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
    api.add_resource(Extract, args.path+'/extract')
    api.add_resource(Lookup, args.path+'/lookup')
    app.run(host=args.host, port=args.port)

    # Free resources
    free_alphabets(OPTIONS)


if __name__ == '__main__':
    main()
