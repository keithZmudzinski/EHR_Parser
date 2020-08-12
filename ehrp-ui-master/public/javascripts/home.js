angular.module('ehrpui').controller('homeController', function ($scope, $http, $timeout) {

  $scope.medTerm = null;
  $scope.medText = null;
  $scope.extractResults = null;
  $scope.lookupResults = null;

  $scope.initMedText = function() {
    // $scope.medText= "87 yo F with h/o CHF, COPD on 5 L oxygen at baseline, tracheobronchomalacia s/p stent, presents with acute dyspnea over several days, and lethargy. "+
    // "This morning patient developed an acute worsening in dyspnea, and called EMS. EMS found patient tachypnic at saturating 90% on 5L. "+
    // "Patient was noted to be tripoding. She was given a nebulizer and brought to the ER.\n"+
    // "According the patient's husband, she was experiencing symptoms consistent with prior COPD flares. "+
    // "Apparently patient was without cough, chest pain, fevers, chills, orthopnea, PND, dysuria, diarrhea, confusion and neck pain. "+
    // "Her husband is a physician and gave her a dose of levaquin this morning.\n"+
    // "In the ED, patient was saturating 96% on NRB. CXR did not reveal any consolidation. Per report EKG was unremarkable. "+
    // "Laboratory evaluation revealed a leukocytosis if 14 and lactate of 2.2. Patient received combivent nebs, solumedrol 125 mg IV x1, aspirin 325 mg po x1. "+
    // "Mg sulfate 2 g IV x1, azithromycin 500 mg IVx1, levofloxacin 750 mg IVx1, and Cefrtiaxone 1g IVx1."+
    // "Patient became tachpnic so was trialed on non-invasive ventilation but became hypotensive to systolics of 80, so noninvasive was removed and patient did well on NRB and nebulizers for about 2 hours. "+
    // "At that time patient became agitated, hypoxic to 87% and tachypnic to the 40s, so patient was intubated. Post intubation ABG was 7.3/60/88/31. "+
    // "Propafol was switched to fentanyl/midazolam for hypotension to the 80s. Received 2L of NS. On transfer, patient VS were 102, 87/33, 100% on 60% 450 x 18 PEEP 5. "+
    // "Patient has peripheral access x2.\nIn the ICU, patient appeared comfortable.";

    $scope.medText= "This 68 yo female presents with intermittent dizziness, nausea, vomiting, and diarrhea of about 1 week duration. "+
    "Over this period of time, she has been unable to take in any significant PO intake without vomiting. Her dizziness and lightheadedness are most notable when she stands up, and she has difficulty maintaining her balance due to this. "+
    "She also notes that has been very tired for this past week, spending approximately 20 hours per day in bed sleeping. "+
    "She denies pain, headache, fevers, chills, SOB, chest pain, hematemesis, bloody stool, tarry stool, dysuria, hematuria, and increased bleeding or bruising. "+
    "The patient is unable to provide further details or further describe her symptoms, and has no idea what might be causing them. "+
    "She does deny any recent sick contacts, eating any new or abnormal foods, eating any potentially raw meats, and drinking large amounts of tonic water, or anything else that contains quinine. "+
    "\n\nCURRENT MEDICATIONS:"+
    "\nLasix, 20 mg PO daily"+
    "\nPotassium Chloride, 8 meq PO daily"+
    "\nAtenolol, 50 mg PO daily"+
    "\nLipitor, 10 mg PO daily"+
    "\nNorvasc, 5 mg PO daily";

    $scope.medTerm = "Hypertension";
  }

  $scope.extract = function() {
    var query = '';

    // Set medText to empty string if not defined, otherwise remove surrounding whitespace
    var medtext = ($scope.medText === undefined)? '' : String($scope.medText).trim();

    // Handle MS Word encodings of single and double quotes
    var medtext = medtext.replace(/[\u2018\u2019]/g, "'").replace(/[\u201C\u201D]/g, '"');

    // Remove surrounding whitespace
    var postdata = {"text" : medtext.trim()};

    // Set URI to call the post method of class Extract in ehrp-api-master/ehr_api.py
    var uri = getServer()+'/ehrs';
    console.log('Public: '+uri);

    $scope.extractResults = [];
    $http.post(uri, postdata).then(function(response) {
      if (response.status === 200) {

        // Get data returned from extract_concepts
        const extractResults = response.data.records;
        let numResults = 0;
        let variableName = ''

        // Reset displayed tables
        for (var property of Object.keys($scope)) {
            if (property.slice(0, 8) === '_extract') {
                $scope[property] = null;
            }
        }

        // Break list of different types into distinct variables
        for (const result of extractResults) {

            // Change ontology names  to uppercase
            for (var instance of result.instances) {
              if (instance.onto) {
                instance.onto = instance.onto.toUpperCase();
              }
            }

            // Create angular variable
            // normalize result.name, replacing all whitespace with underscores
            variableName = result.name.replace('/\s/g', '_').toLowerCase();

            // Uppercase just the first letter of the variable name
            variableName = variableName.toUpperCase().charAt(0) + variableName.slice(1);

            // To differentiate between extract results and lookup results
            variableName = `_extract${variableName}`;
            $scope[variableName] = result;
            numResults += result.instances.length;
        }

        // Create message in response to search
        numberOfTypes = response.data.count;
        var searchMessage = `${numResults} records retrieved, spread over ${numberOfTypes} types.`;
        $scope.extractMsg = searchMessage;

        console.log('Public: Success ' + searchMessage);

        // Make message visible in home.ejs
        $('#extract-results').removeClass('hidden');

      // If response not 200
      } else {
        console.log('Public: ' + response.data.error);
        $scope.searchMsg = "Failed!! Check query and try again.";
      }
    });
  }

  $scope.lookup = function() {
    var query = '';
    var medterm = ($scope.medTerm === undefined)?'':String($scope.medTerm).trim();
    // var uriparams = encodeURIComponent('query='+query.trim()+'&count='+count);
    var uriparams = 'term=' + medterm.trim();
    var uri = getServer()+'/terms?'+uriparams;
    console.log('Public:'+uri);
    $scope.lookupResults = [];
    $http.get(uri).then(function(response) {
        if (response.status === 200) {
          // Get data returned from extract_concepts
          const lookupResults = response.data.records;
          let numResults = 0;
          let variableName = ''

          // Reset displayed tables
          for (var property of Object.keys($scope)) {
              if (property.slice(0, 7) === '_lookup') {
                $scope[property] = null;
              }
          }

          // Break list of different types into distinct variables
          for (const result of lookupResults) {

              // Change ontology names to uppercase
              for (var instance of result.instances) {
                if (instance.onto) {
                  instance.onto = instance.onto.toUpperCase();
                }
              }

              // Create angular variable
              // normalize result.name, replacing all whitespace with underscores
              variableName = result.name.replace('/\s/g', '_').toLowerCase();

              // Uppercase just the first letter of the variable name
              variableName = variableName.toUpperCase().charAt(0) + variableName.slice(1);

              // To differentiate between extract results and lookup results
              variableName = `_lookup${variableName}`;
              $scope[variableName] = result;
              numResults += result.instances.length;
          }

          // Create message in response to search
          numberOfTypes = response.data.count;
          var searchMessage = `${numResults} records retrieved, spread over ${numberOfTypes} types.`;
          $scope.lookupMsg = searchMessage;

          console.log('Public: Success ' + searchMessage);

          // Make message visible in home.ejs
          $('#lookup-results').removeClass('hidden');

        // If response not 200
        } else {
          console.log('Public: ' + response.data.error);
          $scope.searchMsg = "Failed!! Check query and try again.";
        }
    });
  }

});
