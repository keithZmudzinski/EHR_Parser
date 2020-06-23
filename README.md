# Electronic Health Record Parser
This API provides a service to extract medical information from electronic health records (EHRs).<br>

This repository provides code for both the API itself, and an example interface that makes use of the API.
+ The `ehrp_api_master` directory contains the API
+ The `ehrp_ui_master` directory contains the interface

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
2. execute `python ./ehrp-api.py`

#### To make requests to the API
  1. `lookup` functionality is accessible through a `GET` request to the following URL<br>
    URL : http://localhost:8020/ehrp/lookup<br>
    ##### Parameters
    + Required:<br>
      + Name: 'text'
        + Type: string
        + Description: This should be the term you want to lookup.<br>

    ##### Example request:
    + GET: http://localhost:8020/ehrp/lookup?text=hypertension
    + RESPONSE:
    


  2. POST request to provide medical text to process.<br>
    URL : http://localhost:8020/ehrp/extract<br>
    Required parameter: 'text'; this should be the text you want to process.<br>
    Optional parameter: 'types'; this should be a list of strings of types of medical terms you want to be extracted. If not
    provided, all types are tried. If only one type is desired, use a list with just one element.

      Possible values in 'types':
        * 'drug'
        * 'prescription'
        * 'disorder'

      NOTE: Content type should be in JSON format, e.g. if using the `requests` library for python:<br>
    `resp = requests.post('http://localhost:8020/ehrp/extract', json={'text': text, 'types':types_list})`

Both GET and POST requests return JSON objects.
___
### Example Interface
1. install nodejs
2. `cd` into the ehrp-ui-master directory
3. execute `npm install`
4. execute `npm start` to start the local server
5. Visit URL at http://localhost:3020/<br>
Port numbers and REST api can be configured in bin/settings.js file
## Error response codes
* 400: Malformed url; check your base url to see if it conforms to one of the two above.
* 422: Raised if optional parameter has unknown value; check the value(s) you are using for `type` or `types`, make sure it is one of the allowable types listed above.
