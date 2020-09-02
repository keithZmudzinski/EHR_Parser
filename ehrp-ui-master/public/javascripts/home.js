angular.module('ehrpui').controller('homeController', function ($scope, $http, $timeout) {

  $scope.medTerm = null;
  $scope.medText = null;
  $scope.extractResults = null;
  $scope.lookupResults = null;

  $scope.initMedText = function() {
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
          let numResults = 1;
          let variableName = '';

          // Reset displayed tables
          for (var property of Object.keys($scope)) {
              if (property.slice(0, 7) === '_lookup') {
                $scope[property] = null;
              }
          }

          // Break list of different types into distinct variables
          for (const result of lookupResults) {

            // Change ontology names to uppercase
            result.onto = result.onto.toUpperCase();

            // Create angular variable
            // normalize result.name, replacing all whitespace with underscores
            variableName = result.name.replace('/\s/g', '_').toLowerCase();

            // Uppercase just the first letter of the variable name
            variableName = variableName.toUpperCase().charAt(0) + variableName.slice(1);

            // To differentiate between extract results and lookup results
            variableName = `_lookup${variableName}`;
            $scope[variableName] = result;
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
