import re
import sys
import WordSenseDisambiguation as WSD

'''
    Provides the 'get_entry' functionality to get the CUI and ONTO of a term.
'''
class DictionaryParser():
    def __init__(self, simple_words, compound_words):
        self.simple_entities =  simple_words
        self.compound_entities = compound_words
        self.entities = self.simple_entities + self.compound_entities

        self.terms = {}
        self.terms = self.parse_entities()


    def parse_entities(self):
        ''' Create a lookup table of entities, indexed by term '''
        for entity in self.entities:
            # Parse Unitex format dictionary entry
            decomposed_entity = self.decompose(entity)

            # Put relevant information into lookup table
            self.terms[decomposed_entity.term] = decomposed_entity.instances

    def get_entry(self, term, category, context):
        ''' Get CUI and ONTO of 'term' '''
        entity = self.entities[term]

        # If the entity has multiple meanings
        if entity.is_homonym():
            # Decide which meaning is most appropriate, given the context
            instance = WSD.get_meaning(entity.instances, entity.term, context)

            # If most appropriate meaning is the same as current category
            if instance.category == category:
                return instance.cui, instance.onto
            # If most appropriate meaning is different from current category
            else:
                return None, None
        else:
            # entity only has one instance
            instance = entity.instance[0]
            return instance.cui, instance.onto

    def decompose(self, entity):
        ''' Break Unitex format entry into an Entity '''
        lemma_separator = entity.find('.')
        while lemma_separator >= 0:
            # try:
            # A cui is 8 characters long
            cui_start = lemma_separator - 8
            # A homonym lemma is 7 characters long
            homonym_start = lemma_separator - 7

            possible_cui = entity[cui_start:lemma_separator]
            possible_homonym = entity[homonym_start:lemma_separator]

            if is_cui(possible_cui):
                # Term is from the beginning until 1 character before the cui starts
                term = entity[:cui_start-1]
                lemma = possible_cui
                info = entity[lemma_separator+1:]
                break
            elif possible_homonym == 'HOMONYM':
                term = entity[:homonym_start-1]
                lemma = possible_homonym
                info = entity[lemma_separator+1:]
                break
            lemma_separator = entity[lemma_separator:].find('.')

        # Split remaining attributes
        attributes = unescaped_split('\\+', info)

        # First attribute is always category
        category = attributes[0]

        # Remove category from attributes
        attributes = attributes[1:]

        # Aggregate information into 'instances' list
        instances = []

        # If entity has multiple possible meanings, return Entity
        if lemma == 'HOMONYM':
            cuis = []

            for attribute in attributes:
                if is_cui(attribute):
                    cuis.append(attribute)
                    last_attribute_was_cui = True

                elif last_attribute_was_cui:
                    # Use all cuis seen so far, creating an Instance per cui
                    instances.extend( [Instance(cui, attribute, category) for cui in cuis] )
                    cuis = []
                    last_attribute_was_cui = False

                # If neither a CUI or onto, then we've seen all CUIs and ontos for this entity
                else:
                    break

        # If entity has just one meaning, return Instance
        else:
            # If entity is not a homonym, onto follows category
            onto = attributes[0]
            instances = [ Instance(lemma, onto, category) ]

        return  Entity(term, instances)

class Entity:
    def __init__(self, term, instances):
        self.term = term
        self.instances = instances

    def is_homonym(self):
        if len(self.instances) > 1:
            return True
        return False

class Instance:
    def __init__(self, cui, onto, category):
        self.cui = cui
        self.onto = onto
        self.category = category

def unescaped_split(delimiter, line):
    # Only split on unescaped versions of delimiter in line
    return re.split(r'(?<!\\){}'.format(delimiter), line)

def is_cui(attribute):
    starts_with_C = attribute[0] == 'C'
    length_of_eight = len(attribute) == 8

    try:
        all_numbers = [int(char) for char in attribute[1:]]
        all_numbers = True
    except ValueError:
        all_numbers = False

    return starts_with_C and length_of_eight and all_numbers
