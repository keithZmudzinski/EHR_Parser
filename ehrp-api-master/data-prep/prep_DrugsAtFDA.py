import re

def unescape_instances_of(to_remove, input):
    # Remove escaped characters from input used in comparison
    while(True):
        indx = input.find('\\' + to_remove)
        if(indx > -1):
            input = input[:indx] + to_remove + input[indx+2:]
        else:
            break
    return input
    
def remove_blank_lines(file):
    file = open(file, 'r')
    to_output = ''

    for line in file:
        if line != '\n':
            to_output += line
    file.close()
    return to_output

def main():
    dic_file = 'DrugsAtFDAData/DrugsAtFDA.dic'
    data_file = 'DrugsAtFDAData/Products.txt'

    dic = open(dic_file, 'r')
    data = open(data_file, 'r')
    output_string = ''
    names_not_found_lines = ''

    dic_lines = dic.readlines()
    # Exclude column names
    data_lines = data.readlines()[1:]

    for i, dic_line in enumerate(dic_lines):
        # Get info from dictionary
        drug_name, attributes = re.split(r'(?<!\\),', dic_line)
        drug_name_testing = drug_name.upper()
        drug_name_testing = unescape_instances_of(',', drug_name_testing)
        drug_name_testing = unescape_instances_of('.', drug_name_testing)
        drug_name_testing = unescape_instances_of('+', drug_name_testing)

        # Get matching drug in data
        for data_line in data_lines:
            info = data_line.split('\t')
            application_id = info[0]
            data_name = info[5]
            ingredient_name = info[6]

            # If either brand name or active ingredient is found
            if drug_name_testing == data_name or drug_name_testing == ingredient_name:
                break
        # Save error locations and names for manual investigation
        else:
            names_not_found_lines += str(i) + ', ' + drug_name + '\n'
            application_id = ''

        # Append updated drug information
        output_string += '{},{}{}'.format(drug_name, application_id, attributes)

    # Save updated dictionary
    to_save = open('DrugsAtFDAData/updated_but_missing_some.dic', 'w')
    to_save.write(output_string)
    to_save.close()

    # Save found errors
    errors = open('DrugsAtFDAData/errors.txt', 'w')
    errors.write(names_not_found_lines)
    errors.close()
