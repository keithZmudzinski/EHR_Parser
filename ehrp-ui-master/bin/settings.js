'use strict';

console.log('Loading settings..');

const SETTINGS = {
    // WEB_SERVICES_URI : 'http://t3610.istb4.dhcp.asu.edu:8020/',     //api link
    WEB_SERVICES_URI : 'http://localhost:8020/ehrp/',     //api link
    SERVER_PORT : 3020                                              //port for nodejs
};

console.log('App available on : http://localhost:'+ SETTINGS.SERVER_PORT + "/");
console.log('Connected to REST API : '+ SETTINGS.WEB_SERVICES_URI);

module.exports = SETTINGS;
