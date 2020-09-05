#!/bin/bash

DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing redis and node
echo [$(date +"%F %T")] ================================

brew install redis node python@3.8 ffmpeg
export PATH="/usr/local/opt/python@3.8/bin:$PATH"

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing python dependencies
echo [$(date +"%F %T")] ================================

python3.8 -m pip install -U pip
python3.8 -m pip install --user -U --ignore-installed -r ${DIR}/requirements.txt

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing node packages
echo [$(date +"%F %T")] ================================

npm install --max_old_space_size=8000

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Compiling source code
echo [$(date +"%F %T")] ================================

node_modules/.bin/webpack --config webpack.config.js --mode=production

# . ${DIR}/setup_local_dir.sh

echo [$(date +"%F %T")] Add /usr/local/opt/python@3.8/bin to PATH in default bash config
echo [$(date +"%F %T")] Or export PATH=\"/usr/local/opt/python@3.8/bin:\$PATH\" in the current shell