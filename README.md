# Electronic Health Record Parser
This API provides a service to extract medical information from electronic health records (EHRs).<br>

This repository provides code for both the API itself, and an example interface that makes use of the API.
+ The `ehrp_api_master` directory contains the API
+ The `ehrp_ui_master` directory contains the interface
+ The `graph_sources` directory contains the source information for the language graphs used in the api, in Unitex format.

## API
### Requirements
+ [python-unitex](https://github.com/patwat/python-unitex)
+ Flask
+ flask-restful
+ python 3

### Usage
The API provides two main functions; `extract` and `lookup`. `extract` is used to pull and return medical information from a user-provided piece of text. `lookup` is used to return medical information about a user-provided search term.

#### To start the API
1. `cd` into the ehrp-api-master directory
2. execute `python3 ./ehrp_api.py`

___
#### To make requests of the API

  #### Lookup
  + `lookup` functionality is accessible through a `GET` request to the following URL<br>
    URL : http://localhost:8020/ehrp/lookup<br>
    ##### Parameters
    + Required:<br>
      + Name: 'text'
        + Type: string
        + Description: This should be the term you want to lookup.<br>

    ##### Example request:
    + GET: http://localhost:8020/ehrp/lookup?text=hypertension
    + RESPONSE:<br>
    ```
    [
        {
          'instances': [
            {'onto': 'Meddra', 'term': 'hypertension', 'umid': '10020772'}
          ],
          'name': 'lookup'
        }
    ]
    ```
___

#### Extract
  + `extract` functionality is accessible through a `POST` request to the following URL<br>
    URL : http://localhost:8020/ehrp/extract
    ##### Parameters
    + Optional:
      + Name: 'text'
        + Type: string
        + Description: This should be the text you want to process.
        + **NOTE**: Exactly one of 'text' and 'file' can be used in the same request. If both are used, or neither are used, a 422 error response will be returned.
      + Name: 'types'
        + Type: string
        + Description: This parameter specifies which types of medical data you want to be extracted from the text. Multiple types can be specified, all using 'types' as a parameter name.
        + Example using the python requests library:<br>
        ```
        args = {
          'text': medical_text_string,
          'types': ['drug', 'pt_summary', 'ami']
        }
        response = requests.post('http://localhost:8021/ehrp/extract', data=args)
        ```
        + Possible values for 'types':  
          + 'drug': Looks for drug names
          + 'prescription': Looks for drug names in conjunction with a dosage, e.g. '350 mg of Tylenol taken as needed'
          + 'disorder': Looks for disorder names
          + 'chf': Looks for terms related to congestive heart failure
          + 'ami': Looks for terms related to acute myocardial infarction
          + 'pna': Looks for terms related to pneumonia
          + 'comorbidity': Looks for other related afflictions aside from chf, ami, or pna
          + 'pt_summary': Looks for the age and gender of the patient
        + If 'types' is not specified, all types will be used.
        + Currently, if more than three types are desired, it is faster to not specify any types and let the API process all types, than it is to specify 4+ types. This just means the user will have to extract the specific types they want from the returned JSON object.
      + Name: 'file'
        + Type: file object
        + Description: Allows a text file to be uploaded in place of the 'text' parameter. The content of the uploaded file is then parsed, the same way 'text' is.
        + **NOTE**: Exactly one of 'text' and 'file' can be used in the same request. If both are used, or neither are used, a 422 error response will be returned.
        + Example using the python requests library:
        ```
        args = {
          'types': ['drug', 'pt_summary', 'ami']
        }
        file = open('medical_text_path', 'rb')
        response = requests.post('http://localhost:8021/ehrp/extract', data=args, files={'file': file))
        ```

Both GET and POST requests return JSON objects.

### Error response codes
* 400: Malformed url; check your base url and parameter names to see if they conform to the descriptions above.
* 422: Two possible reasons:
  1. 'types' parameter has unknown value; check the value(s) you are using for 'types', make sure it is one of the allowable types listed above.
  2. In a POST request to the `extract` functionality, either both 'text' and 'file' have been specified, or neither have been specified.




___
## Example Interface
1. install nodejs
2. `cd` into the ehrp-ui-master directory
3. execute `npm install`
4. execute `npm start` to start the local server
5. Visit URL at http://localhost:3020/<br>
Port numbers and REST api can be configured in bin/settings.js file
