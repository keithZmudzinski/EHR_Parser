import re

''' Creates two dict objects:
        one maps concept names to concept ids,
        one maps concept names to ontology they come from
'''
class DictionaryParser():
    def __init__(self, simple_words, compound_words, ontology_names):
        self.simple_entities =  simple_words
        self.compound_entities = compound_words
        self.entities = self.simple_entities + self.compound_entities

        # ID dictionary: Maps labels to UMIDS
        self.id_dict = {}

        # Ontology dictionary: Maps labels to ontologies
        self.onto_dict = {}

        # Used to determine which ontology an entity came from
        self.ontology_names = ontology_names

    def parse_dictionaries(self):
        for entity in self.entities:
            # This is expected format, need to make dictionary comply with expected format
            parts = re.split(r'(?<!\\),', entity)
            label = parts[0]
            label = label.replace('\\,', ',')
            cid = parts[-1].split('.')[0]

            # Get ontology name based on if ontology name in entity. Should be exactly one ontology name per entity.
            onto = [ ontology for ontology in self.ontology_names if ontology.lower() in entity.lower() ][0]

            # If concept not yet seen, add to id and onto dicts.
            if self.id_dict.get( label.lower() ) == None:
                self.id_dict[label.lower()] = cid
                self.onto_dict[label.lower()] = onto
