#!/usr/bin/env bash
redis-server &
node app/dist/js/main.js --config ./data/config.yml