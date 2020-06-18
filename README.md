# Electronic Health Record Parser
Service to extract information from raw EHR documents.

## Usage
### UI
1. install nodejs
2. `cd` into the ehrp-ui-master directory
3. execute `npm install`
4. execute `npm start` to start the local server
5. Visit URL at http://localhost:3020/<br>
Port numbers and REST api can be configured in bin/settings.js file
___
### Flask Server
#### Requirements
* [python-unitex](https://github.com/patwat/python-unitex)
* Flask
* flask-restful

Provides two APIs:
  1. GET request to lookup specific medical terms.<br>
    URL : http://localhost:8020/ehrp/lookup?text=hypertension<br>
    Required parameter: 'text'; this should be the term you want to lookup.<br>
    Optional parameter 'type': The type of term you are looking up. If not provided, all types are tried.
    
      Possible values for 'type':
        * 'drug'
        * 'prescription'
        * 'disorder'

  2. POST request to provide medical text to process.<br>
    URL : http://localhost:8020/ehrp/extract<br>
    Required parameter: 'text'; this should be the text you want to process.<br>
    Optional parameter: 'types'; this should be a list of strings of types of medical terms you want to be extracted. If not
    provided, all types are tried.
    
      Possible values for 'type':
        * 'drug'
        * 'prescription'
        * 'disorder'
        
    NOTE: Content type should be in JSON format, e.g. if using the `requests` library for python:<br>
    `resp = requests.post('http://localhost:8020/ehrp/extract', json={'text': text, 'types':type})`

Both GET and POST requests return JSON objects.
___
## Error response codes
* 400: Malformed url; check your base url to see if it conforms to one of the two above.
* 422: Raised if optional parameter has unknown value; check the value(s) you are using for `type` or `types`, make sure it is one of the allowable types listed above.
