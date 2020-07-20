import os
import re


# Get all file names
all_files = os.listdir('../Dictionaries')
# Retain only .dic files
dic_files = [file for file in all_files if os.path.splitext(file)[1] == '.dic']

# For building up dlf and dlc files
multiple_words = ''
single_words = ''

for file_path in dic_files:
    input_dic = open(os.path.join('../Dictionaries', file_path), 'r')

    # For every entry in a dictionary
    for line in input_dic:
        # Get the term for that line, splitting on non-escaped commas
        term, attributes = re.split(r'(?<!\\),', line)

        # If the term consists of multiple words
        if ' ' in term:
            multiple_words += line
        # If term is just one word
        else:
            single_words += line

# Create separate dictionaries with single and multiple words
single_word_output = open('dlf', 'w')
multiple_word_output = open('dlc', 'w')

# Write the output
single_word_output.write(single_words)
multiple_word_output.write(multiple_words)

# Close the newly created dictionaries
single_word_output.close()
multiple_word_output.close()
