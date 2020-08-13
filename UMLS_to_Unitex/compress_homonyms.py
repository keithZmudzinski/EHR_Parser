import sys
import re
import os

def unescaped_split(to_split, delimiter):
    format_string = r'(?<!\\){}'.format(delimiter)
    return re.split(format_string, to_split)

def is_cui(to_check):
    to_check = to_check.strip()
    if to_check[0] == 'C' and (len(to_check) == 8):
        for entry in to_check[1:]:
            if ('1234567890'.find(entry)) == -1:
                return False
        return True
    return False

def is_tui(to_check):
    to_check = to_check.strip()
    if to_check[0] == 'T' and (len(to_check) == 4):
        for entry in to_check[1:]:
            if ('1234567890'.find(entry) == -1):
                return False
        return True
    return False

def format_homonyms(input_files, output_folder):
    CATEGORIES = [
        'Drug',
        'Device',
        'Procedure',
        'Disorder'
    ]

    for file_path in input_files:
        print('Formatting homonyms in', file_path)

        input_file = open(file_path, 'r')

        output_file_name = file_path.split('_')[1]
        output_file_path = os.path.join(output_folder, output_file_name)
        output_file = open(output_file_path, 'w')

        # Do this for all dictionaries
        corrected_dic = ''
        for count, line in enumerate(input_file):
            if count % 100000 == 0:
                print('On line', count)
            # Split the dictionary entry to get the lemma,
            #   We only want to format homonyms
            term, info = unescaped_split(line, ',')
            lemma, info = unescaped_split(info, '\.')

            if lemma == 'HOMONYM':

                # Used for holding dictionary entry information
                category = []
                tuis = []
                type_names = []
                ontos_and_cuis = {}

                # Get just the semantic information in an array
                semantics = unescaped_split(info, '\+')
                semantics = iter(semantics)

                # Assign each piece of semantic information to appropriate category
                for descriptor in semantics:

                    # Catch categories
                    if descriptor in CATEGORIES:
                        category.append(descriptor)

                    # Catch cuis
                    elif is_cui(descriptor):
                        # Ontos always follow cuis
                        onto = next(semantics)
                        try:
                            ontos_and_cuis[onto].append(descriptor)
                        except KeyError:
                            ontos_and_cuis[onto] = [descriptor]

                    # Catch tuis
                    elif is_tui(descriptor):
                        tuis.append(descriptor.strip())

                    # Catch semantic type names
                    else:
                        type_names.append(descriptor.strip())

                # Format semantic information into one string
                formatted_categories = '+'.join(category)

                formatted_ontos_and_cuis = []
                for onto, cuis in ontos_and_cuis.items():
                    formatted_ontos_and_cuis.append('+'.join(cuis) + '+{}'.format(onto))
                formatted_ontos_and_cuis = '+'.join(formatted_ontos_and_cuis)

                formatted_types = '+'.join(type_names)

                formatted_tuis = '+'.join(tuis)

                formatted_semantics = '+'.join([formatted_categories, formatted_ontos_and_cuis, formatted_types, formatted_tuis])

                line = '{},{}.{}\n'.format(term, lemma, formatted_semantics)

            corrected_dic += line

        input_file.close()
        output_file.write(corrected_dic)
        output_file.close()

if __name__ == '__main__':
    main()
