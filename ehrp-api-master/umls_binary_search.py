from APIProcess import APIProcess
import time
import requests

dlc_path = 'resources/Dictionaries/Internal_api_use/dlc'
extract_url = 'http://localhost:8020/ehrp-api/v1/ehrs'
text = '''This 65 y.o. 3- man presents with intermittent dizziness, nausea, vomiting, headaches, bleeding  and diarrhea of about 1 week duration. Over this period of time, she has been unable to take in any significant PO intake without vomiting. Her dizziness and lightheadedness are most notable when she stands up, and she has difficulty maintaining her balance due to this. She also notes that has been very tired for this past ambulance week, spending approximately 20 hours per day in bed sleeping. She denies pain, headache, fevers, chills, chest pain, hematemesis, bloody stool, tarry stool, dysuria, Hematuria, and increased bleeding or bruising. The patient is unable to provide further details or further describe her symptoms, and has no idea what might be causing them. She does deny any recent sick contacts, eating any new or abnormal foods, eating any potentially raw meats, and drinking large amounts of tonic water, or anything else that contains quinine.'''
args = {
    # 'text': ['text', 'text', 'text']
    'text': [text, text, text]
}

class Result:
    def __init__(self, type):
        self.failed = True
        self.succeeded = False
        if type == 'success':
            self.failed = False
            self.succeeded = True

def problem_binary_search(dlc_lines):
    # Stop condition
    if len(dlc_lines) == 1:
        return dlc_lines

    # Split the dlc into left and right halves
    left_lines, right_lines = split_dlc(dlc_lines)

    # Make query on left half
    left_resp = make_query(left_lines)
    if left_resp.failed:
        return problem_binary_search(left_lines)

    # Make query on right half
    right_resp = make_query(right_lines)
    if right_resp.failed:
        return problem_binary_search(right_lines)

def get_initial_dlc_lines():
    file_path = '../UMLS_to_Unitex/Categorized_Dictionaries/dlc'
    with open(file_path, 'r') as dlc_lines:
        return dlc_lines.readlines()

def split_dlc(lines):
    # Get midpoint index
    midpoint = len(lines) // 2

    # Split total lines in half
    left = lines[:midpoint]
    right = lines[midpoint:]

    return left, right

def make_query(lines):
    # Start API each time
    process = APIProcess(1)
    process.start()

    # Wait for the API to start
    time.sleep(2.6)

    # Update dlc file with lines
    with open(dlc_path, 'w') as dlc_file:
        dlc_file.write(''.join(lines))

    # Make query
    resp = False
    try:
        resp = requests.post(extract_url, data=args)

        if resp.status_code == 200:
            resp = True

    except ValueError:
        pass

    except requests.exceptions.ConnectionError as err:
        print(err)
        pass

    # End process that is running API each time
    process.terminate()

    # Result type dependent on response status code
    result = 'success' if resp else 'failure'

    return Result(result)

def find_problem_entry():
    dlc_lines = get_initial_dlc_lines()
    print(problem_binary_search(dlc_lines))

def make_starting_dic_entry(term_to_use, length):
    term = term_to_use
    if not(term):
        term = 'a' * length

    cui = 'C9999999'
    onto = 'TEST'
    category = 'Drug'

    entry = '{},{}.{}+{}'.format(term, cui, category, onto)
    return entry

def entry_decomp(string):
    term, info = string.split(',')
    cui, info = info.split('.')
    category, onto = info.split('+')
    return {
        'term': term,
        'cui': cui,
        'category': category,
        'onto': onto
    }

def length_binary_search(string, low, high):
    if low == high or low-1 == high or low+1 == high:
        return low

    mid = (low + high) // 2
    print('Current mid is\n', mid)

    string_parts = entry_decomp(string)
    term = string_parts['term']
    left_term = term[:mid]
    right_term = term[:high]

    left_string = '{},{}.{}+{}'.format(left_term, string_parts['cui'], string_parts['category'], string_parts['onto'])
    right_string = '{},{}.{}+{}'.format(right_term, string_parts['cui'], string_parts['category'], string_parts['onto'])

    left_response = make_query([left_string])
    if left_response.failed:
        return length_binary_search(string, low, mid)

    right_response = make_query([right_string])
    if right_response.failed:
        return length_binary_search(string, mid+1, high)

def find_proper_length(term_to_use=None, length=1000):
    starting_string = make_starting_dic_entry(term_to_use, length)
    starting_length = len(entry_decomp(starting_string)['term'])-1
    print(length_binary_search(starting_string, 0, starting_length))

problem_terms = {
        'term1': "{30 (calcium ascorbate 25 mg / calcium carbonate 400 mg / cholecalciferol 200 unt / folic acid 2 mg / pyridoxine hydrochloride 25 mg oral tablet [encora am tablet]) / 30 (calcium ascorbate 25 mg / calcium carbonate 600 mg / cholecalciferol 600 unt / folic acid 0.5 mg / pyridoxine hydrochloride 12.5 mg oral tablet [encora pm tablet]) / 60 (linoleic acid 10 mg / omega-3 acid ethyl esters (usp) 650 mg / vitamin e 50 unt oral capsule [encora capsule]) } pack [encora]",

        'term2': "technical component; under certain circumstances\\ a charge may be made for the technical component alone; under those circumstances the technical component charge is identified by adding modifier 'tc' to the usual procedure number; technical component charges are institutional charges and not billed separately by physicians; however\\ portable x-ray suppliers only bill for technical component and should utilize modifier tc; the charge data from portable x-ray suppliers will then be used to build customary and prevailing profiles",

        'term3': "{91 (alpha tocopherol 30 unt / ascorbic acid 60 mg / cholecalciferol 400 unt / copper sulfate 2 mg / docusate sodium 50 mg / folic acid 1 mg / iron carbonyl 90 mg / magnesium oxide 25 mg / pyridoxine hydrochloride 3 mg / retinol acetate 3500 unt / riboflavin 3 mg / thiamine mononitrate 2 mg / vitamin b 12 0.012 mg / zinc sulfate 20 mg oral tablet) / 91 (calcium carbonate 600 mg chewable tablet) } pack",

        'term4': "ca(co)3 600 mg (ca 240 mg) / cuso4 2 mg (cu 0.8 mg) / docusate sodium 50 mg / fe(co)5 90 mg (fe 7.38 mg) / mgo 25 mg (mg 15.075 mg) / vitamin a 3\\500 unt / vitamin b1 2mg / vitamin b2 3 mg / vitamin b6 3 mg / vitamin b9 1 mg / vitamin b12 0.012 mg / vitamin c 60 mg / vitamin d 400 unt / vitamin e 30 unt / znso4 20 mg (zn 4.54 mg) chewable tablet\\ 91 pack count"
    }
if __name__ == '__main__':
    # find_problem_entry()
    find_proper_length(problem_terms['term4'])
    # find_proper_length(problem_terms['term1'][:406])
    # print(len(problem_terms['term1'].split(' ')))
    # print(len(problem_terms['term2'].split(' ')))

    # print(problem_terms['term1'][:406] + '\n')
    # print(problem_terms['term2'][:516]+ '\n')
    # print(problem_terms['term3'][:403]+ '\n')
    # print(problem_terms['term4'][:272]+ '\n')
