#!/bin/bash

echo [$(date +"%F %T")] Making default project dir local-data/scalabel
mkdir -p local-data/scalabel
echo [$(date +"%F %T")] Making default image local serving dir local-data/items
mkdir -p local-data/items
echo [$(date +"%F %T")] Copying default config
cp app/config/default_config.yml local-data/scalabel/config.yml