'''
Biomedical text processing API for EHR Phenotyping
'''

from __future__ import print_function
import argparse
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
from ehrp_utils import load_alphabets, free_alphabets, extract_concepts
from unitex import init_log_system
from unitex.config import UnitexConfig
import yaml

class Extract(Resource):
    '''Extract API for extracting Biomedical named entities and their respective concept IDs'''
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('text', type=str, required=True, location='json',
                               help="text for extracting entities and concept ids")
        self.reqparse.add_argument('types', type=list, required=False, default=None,
                               location='json', help="type of concept to extract")
        super(Extract, self).__init__()

    def post(self):
        '''POST method'''
        print("POST - Extract")     # for debugging

        args = self.reqparse.parse_args()
        text = args['text']
        types = args['types']

        if types == None:
            concepts = extract_concepts(OPTIONS, text)
        else:
            concepts = extract_concepts(OPTIONS, text, types)

        return jsonify(concepts)

class Lookup(Resource):
    '''Lookup API for finding the relevant concept ID'''
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('text', type=str, required=True, location='args',
                               help="text for finding concept id")
        self.reqparse.add_argument('type', type=str, required=True, location='args',
                                help='Type of concept to lookup')
        super(Lookup, self).__init__()

    def get(self):
        '''GET method'''
        print("GET - Lookup")         # for debugging

        args = self.reqparse.parse_args()
        text = args['text']
        type = args['type']

        if type == None:
            concepts = extract_concepts(OPTIONS, text)
        else:
            concepts = extract_concepts(OPTIONS, text, [type])

        return jsonify(concepts)

def main():
    '''Main method : parse arguments and start API'''
    global OPTIONS
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

    print("Starting app . . .")
    app = Flask(__name__)
    api = Api(app)

    # Handle missing page 404 error
    @app.errorhandler(404)
    def page_not_found(error):
        '''Error message for page not found'''
        return "page not found : " + error

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
