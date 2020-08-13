import re
import sys
import gc

def create_homonyms(input_path, output_path):
    print('Creating homonym entries...')
    input_file = open(input_path, 'r')
    output_file = open(output_path, 'w')

    # Used to detect homonyms
    term_cache = {}

    for count, line in enumerate(input_file):
        if count % 500000 == 0:
            print('On line', count)

        # Get the term we are looking at
        try:
            term, info = unescaped_split(',', line)
        except ValueError as err:
            # Special case, where term ends with a slash
            line = line.replace('18\\,', '18,')
            term, info = unescaped_split(',', line)

        # Get CUI and ontology of term in line
        cui, info = unescaped_split('\\.', info)

        info = info.split('+')
        onto = info[0]
        types = info[1:]

        # Get info about previously found instance of term, if there is one
        duplicate_term = term_cache.get(term)
        if duplicate_term:
            if not(cui in duplicate_term['cuis']):
                # If term refers to new concept, save concept and corresponding onto
                duplicate_term['cuis'].append(cui)
                duplicate_term['ontos'].append(onto)
                for type in types:
                    duplicate_term['types'].add(type.strip())
            else:
                output_file.write(line)
        else:
            term_cache[term] = {
                'cuis': [cui],
                'ontos': [onto],
                'types': set()
            }
            for type in types:
                # Remove possibly included endlines
                term_cache[term]['types'].add(type.strip())
            output_file.write(line)
    input_file.close()

    homonyms = ''
    for term, term_info in term_cache.items():
        # If multiple cuis assigned to one term, term is a homonym
        if len(term_info['cuis']) > 1:
            cuis_and_ontos = zip(term_info['cuis'], term_info['ontos'])
            # Place CUI before the Ontology it refers to, and join on '+' symbol
            cuis_and_ontos = '+'.join(['{}+{}'.format(curr_cui, curr_onto) for curr_cui, curr_onto in cuis_and_ontos])
            # Join types with a '+' symbol
            term_types = '+'.join(list(term_info['types']))
            # Put line in unitex format
            homonym = '{},HOMONYM.{}+{}'.format(term, cuis_and_ontos, term_types)
            # Really make sure no extra newlines
            homonym = homonym.replace('\n', '')
            homonyms += homonym + '\n'

    # Close files
    output_file.write(homonyms)
    output_file.close()

def unescaped_split(delimiter, line):
    # Only split on unescaped versions of delimiter in line
    return re.split(r'(?<!\\){}'.format(delimiter), line)
