var express = require('express');
var bodyParser = require('body-parser');
var router = express.Router();
var request = require('request');

router.use(bodyParser.json()); // support json encoded bodies
router.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies

const API_URI = require('../bin/settings').WEB_SERVICES_URI;

router.get('/', function(req, res) {
  try {
    console.log('Route : / from index');
    res.render('home');
  } catch (error) {
    console.log('Route : ' + error);
    res.render('error');
  }
});

router.post('/extract', function(req, res) {
  var result;
  try {
    if (typeof(req.body.text) === 'string') {
      var body = String(req.body.text.trim());
      var uri = API_URI+'extract';
      request.post({
        headers: {'content-type' : 'application/json'},
        url: uri,
        body: JSON.stringify({'text': body})
      }, function (error, response, body) {
        // If we got a response and server had no error
        if (response && response.statusCode === 200) {
          var rawRecords = JSON.parse(body);
          console.log('Route: Success '+ rawRecords.length + ' retrieved.');
          result = {
            status: 200,
            count: rawRecords.length,
            records: rawRecords
          };
        } else if (error) {
          var errorMsg = 'Route: Failed to call API';
          console.error(errorMsg + ":" + error);
          result = {status: 500, error: errorMsg};
        } else {
          var errorMsg = 'Route: Extract failed to retrieve records from services API';
          console.error(errorMsg + ":" + (response)?body:'');
          result = {status: 500, error: errorMsg};
        }
        res.status(result.status).send(result);
      });
    } else {
      var errorMsg = 'Route: Bad extract query ';
      console.warn(errorMsg + ":" + String(req.query.query));
      result = {status: 400, error: errorMsg};
      res.status(result.status).send(result);
    }
  } catch (err) {
    var errorMsg = 'Route: Failed to retrieve records from services API';
    console.error(errorMsg + ":" +err);
    result = {status: 500, error: errorMsg};
    res.status(result.status).send(result);
  }
});

router.get('/lookup', function(req, res) {
  var result;
  console.log('Route: /lookup ' + req.query);
  try {
    if (typeof(req.query.text) === 'string') {
      var query = String(req.query.text.trim());
      var uri = API_URI+'lookup?text='+query;
      console.log('Route: The uri is \''+uri+'\'');
      request.get(uri, function (error, response, body) {
        if (response && response.statusCode === 200) {
          // console.log(body)
          var rawRecords = JSON.parse(body);
          console.log('Route: Success '+ rawRecords.length + ' retrieved.');
          result = {
            status: 200,
            count: rawRecords.length,
            records: rawRecords
          };
        } else if (error) {
          var errorMsg = 'Route: Failed to call API';
          console.error(errorMsg + ":" + error);
          result = {status: 500, error: errorMsg};
        } else {
          var errorMsg = 'Route:Search failed to retrieve records from services API';
          console.error(errorMsg + ":" + (response)?body:'');
          result = {status: 500, error: errorMsg};
        }
        res.status(result.status).send(result);
      });
    } else {
      var errorMsg = 'Route: Bad Search query';
      console.warn(errorMsg + ":" + String(req.query.query));
      result = {status: 400, error: errorMsg};
      res.status(result.status).send(result);
    }
  } catch (err) {
    var errorMsg = 'Route:Failed to retrieve records from services  API';
    console.error(errorMsg + ":" +err);
    result = {status: 500, error: errorMsg};
    res.status(result.status).send(result);
  }
});

// router.get('/search', function(req, res) {
//   var result;
//   console.log('Route: /search ' + req.query);
//   try {
//     if (typeof(req.query.query) === 'string') {
//       var query = String(req.query.query.trim());
//       var count = String(req.query.count.trim());
//       var uri = API_URI+'search?query='+query+'&count='+count;
//       console.log('Route: The uri is \''+uri+'\'');
//       request.get(uri, function (error, response, body) {
//         if (response && response.statusCode === 200) {
//           var rawRecords = JSON.parse(body);
//           console.log('Route: Success '+ rawRecords.records.length + ' retrieved.');
//           result = {
//             status: 200,
//             retrieved: rawRecords.retrieved,
//             available: rawRecords.available,
//             records: rawRecords.records
//           };
//         } else if (error) {
//           var errorMsg = 'Route: Failed to call API';
//           console.error(errorMsg + ":" + error);
//           result = {status: 500, error: errorMsg};
//         } else {
//           var errorMsg = 'Route:Search failed to retrieve records from services API';
//           console.error(errorMsg + ":" + (response)?body:'');
//           result = {status: 500, error: errorMsg};
//         }
//         res.status(result.status).send(result);
//       });
//     } else {
//       var errorMsg = 'Route: Bad Search query';
//       console.warn(errorMsg + ":" + String(req.query.query));
//       result = {status: 400, error: errorMsg};
//       res.status(result.status).send(result);
//     }
//   } catch (err) {
//     var errorMsg = 'Route:Failed to retrieve records from services  API';
//     console.error(errorMsg + ":" +err);
//     result = {status: 500, error: errorMsg};
//     res.status(result.status).send(result);
//   }
// });

module.exports = router;
