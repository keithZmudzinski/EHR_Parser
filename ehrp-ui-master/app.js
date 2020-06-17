var express = require('express');
var path = require('path');
var favicon = require('serve-favicon');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var index = require('./routes/index');
var users = require('./routes/users');

var app = express();

// view engine setup; lets the app know where we store our html
app.set('views', path.join(__dirname, 'views'));
// We use Embedded JS for templating; https://ejs.co/
app.set('view engine', 'ejs');

// Set the icon of the site
app.use(favicon(path.join(__dirname, 'public', 'images/favicon.png')));
// Log information using Morgan
app.use(logger('dev'));
// For parsing json put requests
app.use(bodyParser.json());
// For parsing url encoded get requests
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

// On any method with home URI (get or put), use index.js
app.use('/', index);
// On any method with /users URI (get or put), use users.js
app.use('/users', users);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
  next()
});

module.exports = app;
