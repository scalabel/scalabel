#!/bin/bash

DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

echo [$(date +"%F %T")] Making default project data dir local-data/scalabel
mkdir -p local-data/scalabel
echo [$(date +"%F %T")] Making default image serving dir local-data/items
mkdir -p local-data/items/examples
cp ${DIR}/../examples/cat.webp local-data/items/examples
echo [$(date +"%F %T")] Copying default config
cp ${DIR}/../app/config/default_config.yml local-data/scalabel/config.yml
