// References
// 1. https://codeburst.io/build-a-weather-website-in-30-minutes-with-node-js-express-openweather-a317f904897b

const express = require('express')
const app = express()

app.use(express.static('public'));
app.set('view engine', 'ejs')

app.get('/', function (req, res) {
    res.render('template') // Use this when changing the template to see if it works
    //res.render('introduction') // Otherwise keep it the introduction
})

app.get('/introduction', function (req, res) {
    res.render('introduction')
})

app.get('/authentication', function (req, res) {
    res.render('authentication')
})

app.get('/resources', function (req, res) {
    res.render('resources')
})

app.use('*',function(req, res){
    res.send('Error 404: Not Found!')
})

app.listen(3030, function () {
    console.log('Documentation listening on port 3030!');
    console.log('Visit http://localhost:3030/ to view the documentation.');
})
