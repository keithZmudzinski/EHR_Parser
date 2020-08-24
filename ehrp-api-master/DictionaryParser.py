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

        # Lookup table of terms found in the text
        self.terms = {}
        self.parse_entities()


    def parse_entities(self):
        ''' Create a lookup table of entities, indexed by term '''
        for entity in self.entities:
            # Get an Entity object, after parsing Unitex format dictionary entry
            decomposed_entity = self.decompose(entity)
            # The term itself, ex: 'tylenol'
            term = decomposed_entity.term
            # List of Instance objects
            instances = decomposed_entity.instances

            # If entity is a homonym, put all instances together
            if decomposed_entity.is_homonym:
                try:
                    self.terms[term].extend(instances)
                except KeyError:
                    self.terms[term] = instances
            # If not a homonym, create entry
            else:
                self.terms[term] = instances


    def get_entry(self, term, category, context):
        ''' Get CUI and ONTO of 'term' '''
        instances = self.terms[term.lower()]

        # If the term has multiple meanings
        if len(instances) > 1:
            # Decide which meaning is most appropriate, given the context
            instance = WSD.get_meaning(instances, term, context)

            # If most appropriate meaning is the same as current category
            if instance.category == category:
                return instance.cui, instance.onto
            # If most appropriate meaning is different from current category
            else:
                return None, None
        else:
            # term only has one meaning
            instance = instances[0]
            return instance.cui, instance.onto

    def decompose(self, entity):
        ''' Break Unitex format entry into a list of Instances '''

        # Separate term and lemma from the entity
        term, lemma, info = initial_separate(entity)

        # Boolean indicating if the entity is a homonym or not
        is_homonym = True if lemma == 'HOMONYM' else False

        # Split remaining attributes
        attributes = unescaped_split('\\+', info)

        # First attribute is always category
        category = attributes[0]

        # Remove category from attributes
        attributes = attributes[1:]

        # Aggregate information into 'instances' list
        instances = []

        # If entity has multiple possible meanings
        if is_homonym:
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

        # If entity has just one meaning
        else:
            # If entity is not a homonym, onto follows category
            onto = attributes[0]
            instances = [ Instance(lemma, onto, category) ]

        return Entity(term, is_homonym, instances)

def initial_separate(entity):
    # Find first '.'
    lemma_separator = entity.find('.')

    # While we can find '.'s
    while lemma_separator >= 0:
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

    return term, lemma, info

class Entity:
    def __init__(self, term, is_homonym, instances):
        self.term = term
        self.is_homonym = is_homonym
        self.instances = instances

class Instance:
    def __init__(self, cui, onto, category):
        self.cui = cui
        self.onto = onto
        self.category = category

def is_homonym(instances):
    return True if len(instances) > 1 else False

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
