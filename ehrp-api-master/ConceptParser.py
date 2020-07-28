import os
import sys
from unitex.io import UnitexFile, rm, exists
from DictionaryParser import DictionaryParser
from unitex.tools import UnitexConstants, locate, dico, concord

class ConceptParser:
    ''' Given text, grammar, dictionaries, and a parsing function, will extract
        concepts from text, and return concepts in a dictionary object '''

    # List of all ConceptParser member variables:
    # directory: path of directory holding Unitex files for current parse job
    # options: dictionary holding options for Unitex functions
    # text: path of the text needing concept extraction
    # alphabet_unsorted: path of the alphabet being used, unsorted
    # alphabet_sorted: path of the alphabet being used, sorted
    # grammar: path of the grammar to apply to text
    # dictionaries: list of paths to dictionaries to apply to text
    # parsing_function: input as name of parsing function to use, changed to function pointer to function of same name
    # ontology_names: strictly the names of the ontologies being used, no file extensions or paths attached
    # index: file path of the index created by ConceptParser.locate_grammar()

    def __init__(self, **kwargs):
        # Update object member variable with passed in arguments.
        self.__dict__.update(kwargs)

        self.index = ''

    def setup(self):
        ''' Initalize attributes that may not be created during __init__ '''
        try:
            # Change function name string to function pointer
            self.parsing_function = ConceptParser.__dict__[self.parsing_function]
        except AttributeError:
            sys.stderr.write('ConceptParser.setup requires ConceptParser.parsing_function attribute to be set')

    def parse(self):
        ''' Apply given dictionaries, grammar, and parsing function to text. Return dictionary of found concepts. '''

        # Apply dictionaries
        #   Creates two files of interest
        #   dlf: dictionay of simple words in text
        #   dlc: dictionay of compound words in text
        self.apply_dictionaries()

        # Create an index (File with locations of strings matching grammar)
        self.index = self.locate_grammar()

        # Build concordance (File with actual strings matching grammar)
        self.build_concordance()

        # Get words that are both in text and dictionary
        # dlf file holds dictionary of simple words that are in dictionaries
        simple_words = "%s%s" % (UnitexConstants.VFS_PREFIX, os.path.join(self.directory, "dlf"))
        # dlc file holds dictionary of compound words that are in dictionaries
        compound_words = "%s%s"%(UnitexConstants.VFS_PREFIX, os.path.join(self.directory, "dlc"))

        # Parse all entities that matched in any dictionary
        dictionary_parser = DictionaryParser(self.get_text(simple_words), self.get_text(compound_words), self.ontology_names)
        dictionary_parser.parse_dictionaries()

        # Assign dictionaries
        id_dict = dictionary_parser.id_dict
        onto_dict = dictionary_parser.onto_dict

        # Get contexts
        contexts_file_path = "%s%s"%(UnitexConstants.VFS_PREFIX, os.path.join(self.directory, "concord.txt"))
        contexts = self.get_text(contexts_file_path)

        # Use parsing function specific to this grammar-dictionary/dictionaries combo
        parsed_concepts = self.parsing_function(self, contexts, id_dict, onto_dict)

        # Ensure grammars used afterwards only use assigned dictionaries
        rm(simple_words)
        rm(compound_words)

        # NOTE: parsed_concepts has specific format:
        #   {
        #        name: '<name>',
        #        instances: [
        #            {
        #                <nameOfAttribute1>: '<value>',
        #                <nameOfAttribute2>: '<value>',
        #                ...
        #                <nameOfLastAttribute>: '<value>'
        #            },
        #            ...
        #        ]
        #    }
        return parsed_concepts

    def apply_dictionaries(self):
        ''' Creates .dlf and .dlc files holding words in both dictionaries and text. '''
        if self.dictionaries is not None:
            dictionaries_applied_succesfully = dico(self.dictionaries, self.text, self.alphabet_unsorted, **self.options['tools']['dico'])

            if dictionaries_applied_succesfully is False:
                sys.stderr.write("[ERROR] Dictionaries application failed!\n")
                sys.exit(1)
        else:
            sys.stderr.write("[ERROR] No dictionaries specified.\n")

    def locate_grammar(self):
        ''' Return index file path, holding locations of matching instances of grammar in text. '''

        # Locate patterns that match grammar
        grammar_applied_successfully = locate(self.grammar, self.text, self.alphabet_unsorted, **self.options['tools']['locate'])

        # Locate created concord.ind file
        index = "%s%s" % (UnitexConstants.VFS_PREFIX, os.path.join(self.directory, "concord.ind"))

        # If application failed or couldn't find associated file
        if grammar_applied_successfully is False or exists(index) is None:
            sys.stderr.write("[ERROR] Locate failed!\n")

        return index

    def build_concordance(self):
        ''' Create concord file that holds actual text matching grammar in text file. '''

        concordance_built_successfully = concord(self.index, self.alphabet_sorted, **self.options["tools"]["concord"])
        if concordance_built_successfully is False:
            sys.stderr.write("[ERROR] Concord failed!\n")
            sys.exit(1)

    def get_text(self, file_path):
        '''Get text contents from a file'''
        if exists(file_path) is False:
            sys.stderr.write("[ERROR] File not found\n")
        unfile = UnitexFile()
        unfile.open(file_path, mode='r')
        unfile_txt = unfile.read()
        unfile.close()
        return unfile_txt.splitlines()

    def make_concepts_object(self, name):
        return {'name': name, 'instances': []}

# -------------- DEFINE PARSING FUNCTIONS BELOW ----------------
# Each parsing function must return concepts like so:
#   {
#        name: '<name>',
#        instances: [
#            {
#                <nameOfAttribute1>: '<value>',
#                <nameOfAttribute2>: '<value>',
#                ...
#                <nameOfLastAttribute>: '<value>'
#            },
#            ...
#        ]
#    }
# Dictionary has one 'name' attribute and an 'instances' attribute that holds a
#   list of found concepts.
#   'name' is used for variable references, should be one word.
#   Each instance in instances is a dictionary of desired attributes and values.
#   Each instance has the same set of attributes, but of course different values.
#
# The id_dict and onto_dict dictionaries are dictionaries that associate concepts
#   to their IDs and the ontology that ID comes from. Each key is in all lowercase
#   letters.

    # masterParser is exception to schema of returned dictionaries
    # It returns a list of dictionaries, each dictionary made by a different parsing function
    # There is a try/except block in extract_concepts to handle this case.
    def masterParser(self, contexts, id_dict, onto_dict):
        used_concepts = {}
        parsed_concepts = []

        # Separate contexts by the parsing function they specify
        for context in contexts:
            left_context, output, right_context = context.split('\t')
            to_parse, parsing_function = output.split('__ParsingFunction__')
            to_parse = left_context + '\t' + to_parse + '\t' + right_context

            # If haven't seen parsing_function yet, add it
            if not(used_concepts.get(parsing_function, False)):
                used_concepts[parsing_function] = [to_parse]
            # Otherwise, add to_parse to parsing function's list
            else:
                used_concepts[parsing_function].append(to_parse)

        # Apply appropriate parsing function to list of contexts
        for parsing_function_str, context_list in used_concepts.items():
            parsing_function = ConceptParser.__dict__[parsing_function_str]
            concepts = parsing_function(self, context_list, id_dict, onto_dict)
            parsed_concepts.append(concepts)

        return parsed_concepts

    # {
    #     name: lookup,
    #     instances: [
    #         {
    #             term: '',
    #             umid: '',
    #             onto: '',
    #         }
    #     ]
    # }
    def lookupParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('lookup');

        # What the user provided
        raw_term = contexts[0].strip()

        # Used to match against dictionaries
        term = raw_term.lower()

        # Find term in dictionaries
        try:
            id = id_dict[term]
            onto = onto_dict[term]

            # Save term
            concepts['instances'].append({
                'term': raw_term,
                'umid': id,
                'onto': onto
            })
        # If term is not in any dictionary
        except KeyError:
            pass

        return concepts


    # {
    #     name: drug,
    #     instances: [
    #         {
    #             label: '',
    #             umid: '',
    #             onto: '',
    #             context: '',
    #         }
    #     ]
    # }
    def drugParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('drug');

        for context in contexts:
            parts = context.split('\t')
            label = parts[1]

            for concept in label.split('/'):
                try:
                    onto = onto_dict[concept.lower()]
                    umid = id_dict[concept.lower()]
                    concepts['instances'].append({
                        'label': label,
                        'umid': umid,
                        'onto': onto,
                        'context': parts[0] + label + parts[2]
                    })
                except KeyError as kerror:
                    continue

        return concepts

    # {
    #     name: disorder,
    #     instances: [
    #         {
    #             label: '',
    #             umid: '',
    #             onto: '',
    #             context: ''
    #         }
    #     ]
    # }
    def disorderParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('disorder');

        for context in contexts:
            parts = context.split('\t')
            label = parts[1]

            for concept in label.split('/'):
                try:
                    onto = onto_dict[concept.lower()]
                    umid = id_dict[concept.lower()]
                    concepts['instances'].append({
                        'label': label,
                        'umid': umid,
                        'onto': onto,
                        'context': parts[0] + label + parts[2]
                    })
                except KeyError as kerror:
                    continue

        return concepts

    # {
    #     name: prescription,
    #     instances: [
    #         {
    #             drug: '',
    #             dosage: '',
    #             umid: '',
    #             onto: '',
    #             context: ''
    #         }
    #     ]
    # }
    def prescriptionParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('prescription')

        for context in contexts:
            parts = context.split('\t')
            dosage = parts[1]
            dosage_and_full_context = dosage.split('|')[0]
            pre_dosage = dosage.split('|')[1]
            post_dosage = dosage.split('|')[2]

            # Skip if has no pre or post dosage
            if not(pre_dosage or post_dosage):
                continue

            # The dosage and drug for which the prescription was made
            full_dosage, drug = dosage_and_full_context.split('__SeparatE__')

            # Full context surrounding prescription
            context = parts[0] + full_dosage + parts[2]

            try:
                drug_id = id_dict[drug.lower()]
                used_ontology = onto_dict[drug.lower()]
            except KeyError as kerror:
                drug_id = 'NA'
                used_ontology = 'NA'

            concepts['instances'].append({
                'drug': drug,
                'dosage': full_dosage,
                'umid': drug_id,
                'onto': used_ontology,
                'context': context
            })

        return concepts

     # {
    #     name: chf,
    #     instances: [
    #         {
    #             type: '',
    #             trigger: '',
    #             context: ''
    #         }
    #     ]
    # }
    def chfParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('chf')

        for i, context in enumerate(contexts):
            parts = context.split('\t')
            trigger = parts[1]
            trigger, key = trigger.split('__SeparatE__')
            context = parts[0] + trigger + parts[2]

            concepts['instances'].append({
                'type': key,
                'trigger': trigger,
                'context': context
            })

        return concepts

       # {
    #     name: ami,
    #     instances: [
    #         {
    #             type: '',
    #             trigger: '',
    #             context: ''
    #         }
    #     ]
    # }
    def amiParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('ami')

        for i, context in enumerate(contexts):
            parts = context.split('\t')
            trigger = parts[1]
            trigger, key = trigger.split('__SeparatE__')
            context = parts[0] + trigger + parts[2]

            concepts['instances'].append({
                'type': key,
                'trigger': trigger,
                'context': context
            })

        return concepts

    # {
    #     name: pna,
    #     instances: [
    #         {
    #             type: '',
    #             trigger: '',
    #             context: ''
    #         }
    #     ]
    # }
    def pnaParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('pna')

        for context in contexts:
            parts = context.split('\t')
            trigger = parts[1]
            trigger, type = trigger.split('__SeparatE__')
            context = parts[0] + trigger + parts[2]

            concepts['instances'].append(({
                'type': type,
                'trigger': trigger,
                'context': context
            }))

        return concepts

    # {
    #     name: comorbidity,
    #     instances: [
    #         {
    #             type: '',
    #             trigger: '',
    #             context: ''
    #         }
    #     ]
    # }
    def comorbidityParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('comorbidity')

        for context in contexts:
            parts = context.split('\t')
            trigger = parts[1]
            trigger, type = trigger.split('__SeparatE__')
            context = parts[0] + trigger + parts[2]

            concepts['instances'].append(({
                'type': type,
                'trigger': trigger,
                'context': context
            }))

        return concepts

    # {
    #     name: pt_summary,
    #     instances: [
    #         {
    #             trigger: '',
    #             age: '',
    #             gender: '',
    #             context: ''
    #         }
    #     ]
    # }
    def pt_summaryParser(self, contexts, id_dict, onto_dict):
        concepts = self.make_concepts_object('pt_summary')

        for context in contexts:
            parts = context.split('\t')
            trigger = parts[1]
            trigger, info = trigger.split('__SeparatE__')
            age, gender = info.split(',')
            context = parts[0] + trigger + parts[2]

            concepts['instances'].append(({
                'trigger': trigger,
                'age': age,
                'gender': gender,
                'context': context
            }))

        return concepts
