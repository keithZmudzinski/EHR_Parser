#!/bin/bash

#This script gives status checks on the NGINX server and pm2 deamon processes

sudo systemctl status nginx

pm2 status
