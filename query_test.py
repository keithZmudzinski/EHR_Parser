import requests
import pprint
import sys
from urllib.error import HTTPError
import json
import time

text = '''This 65 y.o. man presents with intermittent dizziness, nausea, vomiting, headaches, bleeding  and diarrhea of about 1 week duration. Over this period of time, she has been unable to take in any significant PO intake without vomiting. Her dizziness and lightheadedness are most notable when she stands up, and she has difficulty maintaining her balance due to this. She also notes that has been very tired for this past week, spending approximately 20 hours per day in bed sleeping. She denies pain, headache, fevers, chills, chest pain, hematemesis, bloody stool, tarry stool, dysuria, hematuria, and increased bleeding or bruising. The patient is unable to provide further details or further describe her symptoms, and has no idea what might be causing them. She does deny any recent sick contacts, eating any new or abnormal foods, eating any potentially raw meats, and drinking large amounts of tonic water, or anything else that contains quinine.

CURRENT MEDICATIONS:
Lasix, 20 mg PO daily
Potassium Chloride, 8 meq PO daily
Atenolol, 50 mg PO daily
Lipitor, 10 mg PO daily
Norvasc, 5 mg PO daily'''

text1 = '''[Report de-identified (Safe-harbor compliant) by De-ID v.6.22.07.0]\n\n\nEXAMINATION PERFORMED:\nCT THORAX  WITHOUT CONTRAST   **DATE[Nov 02 07]     0944 HOURS\n\nCLINICAL HISTORY:   \nMultilobar pneumonia. Evaluation for response.\n\nCOMPARISON:   \nPrevious chest CT scan from **DATE[Oct 04 2007].\n\nTECHNIQUE:   \nHelical CT imaging of the chest was obtained without intravenous\nor oral contrast with contiguous 5.0mm thick axial reconstructions\nusing both lung and standard reconstruction algorithms.\n\nFINDINGS:   \nThe areas of patchy airspace consolidation in the left upper lobe\nhave resolved. There has been near complete resolution of the left\nlower lobe pneumonia. The left pleural effusion has resolved.\n\nThere is persistent extensive dense consolidation in the right\nupper lobe which has decreased. The areas of airspace\nconsolidation in the right lower lobe and right middle lobe have\ndecreased.\n\nIMPRESSION:   \n1. PERSISTENT VERY EXTENSIVE CONSOLIDATION IN THE RIGHT UPPER LOBE\n(WITH NORMAL PATENT AIRWAYS). THE DEGREE AND EXTENT OF RIGHT UPPER\nLOBE CONSOLIDATION HAVE DEFINITELY DECREASED FROM THE PREVIOUS\nSTUDY OF **DATE[Oct 04 2007] CONSISTENT WITH RESOLVING PNEUMONIA. THERE ARE\nNO FINDINGS TO INDICATE ENDOBRONCHIAL OBSTRUCTING LESIONS.\n2. INTERVAL DECREASE IN AREAS OF AIRSPACE CONSOLIDATION AND RIGHT\nLOWER LOBE AND RIGHT MIDDLE LOBE.\n3. NEAR COMPLETE RESOLUTION OF AREAS OF CONSOLIDATION IN LEFT\nUPPER LOBE AND LEFT LOWER LOBE.\n4. RESOLVED LEFT PLEURAL EFFUSION.\n\nMy signature below is attestation that I have interpreted\nthis/these examination(s) and agree with the findings as noted\nabove.\n\nEND OF IMPRESSION:\n\n\n\n\n'''


base_url = 'http://localhost:8020/ehrp/'
extract_url = 'extract'
lookup_url = 'lookup'
file = 'pitt.txt'

def extract(corpus, types=[]):
    args = {}

    # If specific types requested
    if types:
        args['types'] = types

    # Use the text above
    if corpus == 'text':
        args['text'] = text
        query_time = time.time()
        resp = requests.post(base_url+extract_url, data=args)
    # Use pitt.txt file
    else:
        file = open('../English/Corpus/pitt.txt', 'rb')
        query_time = time.time()
        resp = requests.post(base_url+extract_url,files={'file':file}, data=args)
        file.close()

    print('Time taken to get result:', time.time()-query_time)

    print('Query response:', resp.status_code)
    return resp

def lookup(text):
    args = {'text': text}
    resp = requests.get(base_url+lookup_url, args)

    print(resp.status_code)
    return resp

def main():
    args = sys.argv[1:]
    service = args[0]

    # Lookup
    if service == 'lookup':
        term = args[1:]
        term = ' '.join(term)
        resp = lookup(term)

    # Extract
    else:
        # Either text or pitt.txt
        corpus = args[1]
        types = []

        if len(args) > 2:
            types = args[2:]
        resp = extract(corpus, types)

    # Check for errors
    try:
        resp.raise_for_status()
        pprint.pprint(resp.json())
    except HTTPError as err:
        print(err)

if __name__ == '__main__':
    main()
