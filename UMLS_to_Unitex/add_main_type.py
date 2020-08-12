import sys
import os
import re

def overlap(labels, to_label):
    for label in labels:
        if label in to_label:
            return True
    return False

def create_categories(input_path, output_folder):
    print('Creating categories...')
    unlabeled_dic_path = input_path

    output_files = [
        'uncompressed_drugs.dic',
        'uncompressed_disorders.dic',
        'uncompressed_devices.dic',
        'uncompressed_procedures.dic'
    ]
    output_files = [os.path.join(output_folder, file) for file in output_files]

    drug_file = open(output_files[0], 'w')
    disorder_file = open(output_files[1], 'w')
    device_file = open(output_files[2], 'w')
    procedure_file = open(output_files[3], 'w')

    # Define the semantic types that make up each broad category
    drug_labels = [
        'Clinical Drug',
        'Pharmacologic Substance',
        'Antibiotic',
        'Vitamin',
        'Hazardous or Poisonous Substance'
    ]

    disorder_labels = [
        'Disease or Syndrome',
        'Injury or Poisoning',
        'Sign or Symptom',
        'Congenital Abnormality',
        'Virus',
        'Neoplastic Process',
        'Anatomical Abnormality',
        'Acquired Abnormality',
        'Finding'
    ]

    device_labels = [
        'Medical Device',
        'Drug Delivery Device',
        'Research Device'
    ]

    procedure_labels = [
        'Therapeutic or Preventive Procedure',
        'Laboratory Procedure',
        'Diagnostic Procedure',
        'Health Care Activity'
    ]

    input_file = open(unlabeled_dic_path, 'r')
    for count, unlabeled_entry in enumerate(input_file):
        if count % 500000 == 0:
            print('On line {}'.format(count))

        term, info = re.split(r'(?<!\\),', unlabeled_entry)
        cui, types = re.split(r'(?<!\\)\.', info)

        label = []
        file = []
        # Check if entry is a drug
        if overlap(drug_labels, types):
            matched = True
            label.append('Drug')
            file.append(drug_file)

        # Check if entry is a disorder
        if overlap(disorder_labels, types):
            matched = True
            label.append('Disorder')
            file.append(disorder_file)

        # Check if entry is a device
        if overlap(device_labels, types):
            matched = True
            label.append('Device')
            file.append(device_file)

        # Check if entry is a proceture
        if overlap(procedure_labels, types):
            matched = True
            label.append('Procedure')
            file.append(procedure_file)

        if matched:
            # Special case for overlap of Device and Drug label
            #   This occurs when a term is both Drug Delivery Device and a Clinical Drug
            #   In these cases, we want to classify it as a Drug, not both
            if cui != 'HOMONYM' and 'Device' in label and 'Drug' in label:
                label.remove('Device')
            # Make labeled entry
            label = '+'.join(label)
            labeled_entry = '{},{}.{}+{}'.format(term, cui, label, types)

            # Save to file
            for to_write in file:
                to_write.write(labeled_entry)

            # Reset Boolean
            matched = False

    drug_file.close()
    disorder_file.close()
    device_file.close()
    procedure_file.close()
    input_file.close()

if __name__ == '__main__':
    main()
