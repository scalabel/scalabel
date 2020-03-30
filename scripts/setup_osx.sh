#!/bin/bash

DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing redis and node
echo [$(date +"%F %T")] ================================

brew install redis node

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing node packages
echo [$(date +"%F %T")] ================================

npm install

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Compiling source code
echo [$(date +"%F %T")] ================================

node_modules/.bin/webpack --config webpack.config.js --mode=production

. ${DIR}/setup_default_dir.sh