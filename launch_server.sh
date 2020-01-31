#!/usr/bin/env bash

if [[ $1 = '--config' ]]; then
    redis-server &
    node app/dist/js/main.js --config "$2"
else 
    echo "Could not parse command; make sure first argument is --config config/path"
fi
