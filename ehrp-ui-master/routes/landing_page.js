var express = require('express');
var router = express.Router();

//This will be the landing page of PiLabs
//Will contain general info, links, etc...

router.get('/',function(req,res,next){

    res.send(
	    '<h1>Welcome to PiLabs Landing Page!</h1><h2>General Info</h2><h2>Links</h2><a href="/tagger">PiLabs Tagger</a>');
});

module.exports = router;
