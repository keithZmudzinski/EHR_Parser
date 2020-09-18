#!/bin/bash

#This script daemonizes frontend and backend through pm2



#Start API
cd /home/ubuntu/EHR_Parser/ehrp-api-master

pm2 start ehrp_api.py --interpreter python3 --name "backend"


#Start UI
cd /home/ubuntu/EHR_Parser/ehrp-ui-master

pm2 start npm --name "frontend" -- start
