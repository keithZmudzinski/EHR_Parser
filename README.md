# EHR_Parser
Service to extract information from raw EHR documents.

### Usage
#### UI
1. install nodejs
2. `cd` into the ehrp-ui-master directory
3. execute `npm install`
4. execute `npm start` to start the local server
5. Visit URL at http://localhost:3020/<br>
Port numbers and REST api can be configured in bin/settings.js file

#### Flask Server
##### Requirements
* [python-unitex](https://github.com/patwat/python-unitex)
* Flask
* flask-restful

Provides two APIs:
  1. GET request to lookup specific medical terms.<br>
    URL : http://localhost:8020/ehrp/lookup?text=hpertension<br>
    Optional parameter 'type': The type of term you are looking up. If not provided, all types are tried.<br>
        Possible values for type:
        * 'drug'
        * 'prescription'
        * 'disorder'
  2. POST request to provide medical text to process.<br>
    URL : http://localhost:8020/ehrp/extract<br>
    Required parameter: 'text'; this should be the text you want to process<br>
    Optional parameter: 'types'; this should be a list of strings of types of medical terms you want to be extracted. If not
    provided, all types are tried.<br>
        Possible values in types:
        * 'drug'
        * 'prescription'
        * 'disorder'
        
Returns a JSON object


