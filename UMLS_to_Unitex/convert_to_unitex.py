import re
from utility import get_category, unescaped_sub

def umls_to_unitex(conso_path, types_path, output_path):
    print('Converting UMLS to Unitex format...')
    # Get file objects for each file
    conso_file = open(conso_path, 'r')
    types_file = open(types_path, 'r')

    # Create a unitex dict file
    unitex_dict = open(output_path, 'w')

    # Make iterator over file, and start on first line
    types_iter = iter(types_file)
    line = next(types_iter, False)

    # Will store types per CUI, since multiple entries per CUI in MRCONSO
    types_cache = {}
    string_cache = {}

    # Make unitex entry per entry in MRCONSO
    for count, concept_synonym in enumerate(conso_file):
        # Just for tracking purposes
        if count % 500000 == 0:
            print('On line {}'.format(count))

        # Break apart entry in MRCONSO into relavant info
        concept_info = get_info(concept_synonym)

        # Skip entry if we've already seen that term and cui combo before
        if string_cache.get(concept_info['term']) == concept_info['cui']:
            continue
        else:
            string_cache[concept_info['term']] = concept_info['cui']

        # Start unitex entry, setting inflected form = term, lemma = cui, first semantic info = onto
        unitex_entry = '{},{}.{}'.format(concept_info['term'], concept_info['cui'], concept_info['onto'])

        # Try to use cache, if we've already seen this cui before
        types = types_cache.get(concept_info['cui'])

        # If haven't seen this cui, find all types in MRSTY and add to cache
        if types == None:
            # Get types and TUIs
            types, line = get_types(concept_info['cui'], types_iter, line)
            types = unescaped_sub(',', '\\,', types)
            types = types.replace('&#x7C;', '|')
            types_cache[concept_info['cui']] = types

        # Finish unitex entry
        unitex_entry += types + '\n'

        # Only write entry if it is a category of interest
        if get_category(types):
            # Save to disk
            unitex_dict.write(unitex_entry)

    conso_file.close()
    types_file.close()
    unitex_dict.close()

def get_info(conso_line):
    parts = conso_line.split('|')
    info = {
        'cui': parts[0],
        'onto': parts[11],
        'term': parts[14]
    }
    info['onto'] = info['onto'].replace('.', '\\.')
    info['term'] = info['term'].replace(',', '\\,')
    info['term'] = info['term'].replace('&#x7C;', '|')
    return info

def get_types(cui, types, line):
    sem_types = ''
    tuis = ''
    hit_block = False

    while(line):
        parts = line.split('|')
        type_cui = parts[0]
        if cui == type_cui:
            hit_block = True
            sem_types += '+{}'.format(parts[3])
            tuis += '+{}'.format(parts[1])
        elif (cui != type_cui and hit_block):
            break
        line = next(types, False)

    all_types = sem_types + tuis
    return all_types, line