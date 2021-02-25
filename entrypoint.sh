#!/bin/bash

node app/dist/main.js \
    --config /opt/scalabel/local-data/scalabel/config.yml \
    --max-old-space-size=8192
