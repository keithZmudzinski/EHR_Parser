## ehrp-api
Biomedical text processing services API for EHR Phenotyping

Requirements
------------
* [python-unitex](https://github.com/patwat/python-unitex)
* Flask
* flask-restful

API Description
------------
This webservice provides 2 APIs (as of now) with the same functionality.
1. The 'lookup' is a GET request where the input URL encoded which is meant for queries containing shorter texts i.e. concept lookups.
Example Input: 
```
URL : http://localhost:8020/ehrp/lookup?text=hypertension
```
Example Output: 
```
[
    {
        "context": "                                                  \thypertension\t",
        "label": "hypertension",
        "onto": "MEDDRA",
        "umid": "10020772"
    }
]
```

2. The 'extract' uses a POST request where the input contains data in json which has a much larger size limit. Hence, it is suitable for concept extraction from a large chunk of text. 
Example Input:
```
URL : http://localhost:8020/ehrp/extract

Headers:-
'Content-Type':'application/json'

Body:-
{
	"text":"History of Present Illness: 87 yo F with h/o CHF, COPD on 5 L oxygen at baseline, tracheobronchomalacia s/p stent, presents with acute dyspnea over several days, and lethargy. This morning patient developed an acute worsening in dyspnea, and called EMS. EMS found patient tachypnic at saturating 90% on 5L. Patient was noted to be tripoding. She was given a nebulizer and brought to the ER. . According the patient's husband, she was experiencing symptoms consistent with prior COPD flares. Apparently patient was without cough, chest pain, fevers, chills, orthopnea, PND, dysuria, diarrhea, confusion and neck pain. Her husband is a physician and gave her a dose of levaquin this morning."
}
```
Example Output: 
```
[
    {
        "context": "     History of Present Illness: 87 yo F with h/o \tCHF\t, COPD on 5 L oxygen at baseline, tracheobronchomalacia s",
        "label": "CHF",
        "onto": "MEDDRA",
        "umid": "10008502"
    },
    {
        "context": "History of Present Illness: 87 yo F with h/o CHF, \tCOPD\t on 5 L oxygen at baseline, tracheobronchomalacia s/p st",
        "label": "COPD",
        "onto": "MEDDRA",
        "umid": "10010952"
    },
    {
        "context": "e, tracheobronchomalacia s/p stent, presents with \tacute dyspnea\t over several days, and lethargy. This morning ",
        "label": "acute dyspnea",
        "onto": "MEDDRA",
        "umid": "10066606"
    },
    {
        "context": "resents with acute dyspnea over several days, and \tlethargy\t. This morning patient developed an acute worsening ",
        "label": "lethargy",
        "onto": "MEDDRA",
        "umid": "10024264"
    },
    {
        "context": "s morning patient developed an acute worsening in \tdyspnea\t, and called EMS. EMS found patient tachypnic at satu",
        "label": "dyspnea",
        "onto": "MEDDRA",
        "umid": "10013963"
    },
    {
        "context": "veloped an acute worsening in dyspnea, and called \tEMS\t. EMS found patient tachypnic at saturating 90% on 5L. Pa",
        "label": "EMS",
        "onto": "MEDDRA",
        "umid": "10014574"
    },
    {
        "context": "ed an acute worsening in dyspnea, and called EMS. \tEMS\t found patient tachypnic at saturating 90% on 5L. Patient",
        "label": "EMS",
        "onto": "MEDDRA",
        "umid": "10014574"
    },
    {
        "context": "e was experiencing symptoms consistent with prior \tCOPD\t flares. Apparently patient was without cough, chest pai",
        "label": "COPD",
        "onto": "MEDDRA",
        "umid": "10010952"
    },
    {
        "context": "prior COPD flares. Apparently patient was without \tcough\t, chest pain, fevers, chills, orthopnea, PND, dysuria, ",
        "label": "cough",
        "onto": "MEDDRA",
        "umid": "10011224"
    },
    {
        "context": "OPD flares. Apparently patient was without cough, \tchest pain\t, fevers, chills, orthopnea, PND, dysuria, diarrhe",
        "label": "chest pain",
        "onto": "MEDDRA",
        "umid": "10008479"
    },
    {
        "context": "ly patient was without cough, chest pain, fevers, \tchills\t, orthopnea, PND, dysuria, diarrhea, confusion and nec",
        "label": "chills",
        "onto": "MEDDRA",
        "umid": "10008531"
    },
    {
        "context": "nt was without cough, chest pain, fevers, chills, \torthopnea\t, PND, dysuria, diarrhea, confusion and neck pain. ",
        "label": "orthopnea",
        "onto": "MEDDRA",
        "umid": "10031122"
    },
    {
        "context": "out cough, chest pain, fevers, chills, orthopnea, \tPND\t, dysuria, diarrhea, confusion and neck pain. Her husband",
        "label": "PND",
        "onto": "MEDDRA",
        "umid": "10035637"
    },
    {
        "context": "ough, chest pain, fevers, chills, orthopnea, PND, \tdysuria\t, diarrhea, confusion and neck pain. Her husband is a",
        "label": "dysuria",
        "onto": "MEDDRA",
        "umid": "10013990"
    },
    {
        "context": "st pain, fevers, chills, orthopnea, PND, dysuria, \tdiarrhea\t, confusion and neck pain. Her husband is a physicia",
        "label": "diarrhea",
        "onto": "MEDDRA",
        "umid": "10012727"
    },
    {
        "context": "evers, chills, orthopnea, PND, dysuria, diarrhea, \tconfusion\t and neck pain. Her husband is a physician and gave",
        "label": "confusion",
        "onto": "MEDDRA",
        "umid": "10010300"
    },
    {
        "context": " orthopnea, PND, dysuria, diarrhea, confusion and \tneck pain\t. Her husband is a physician and gave her a dose of",
        "label": "neck pain",
        "onto": "MEDDRA",
        "umid": "10028836"
    },
    {
        "context": "resent Illness: 87 yo F with h/o CHF, COPD on 5 L \toxygen\t at baseline, tracheobronchomalacia s/p stent, present",
        "label": "oxygen",
        "onto": "RXNORM",
        "umid": "7806"
    },
    {
        "context": "Her husband is a physician and gave her a dose of \tlevaquin\t this morning.",
        "label": "levaquin",
        "onto": "RXNORM",
        "umid": "217992"
    }
]
```

Utility files
------------

The directory 'data-prep' includes two sample files that help in creating the dictionary.

1. prep_meddra.py : Uses the MEDDRA files to create a dictionary file compatible with Unitex.

2. prep_rxnorm.py : Uses the RXNCONSO.RRF file in RxNorm from UMLS to create a dictionary file compatible with Unitex.

Once the dictionary file has been created, it can be imported into Unitex to transform it from readable (.dic) file into a binary (.bin) file.
