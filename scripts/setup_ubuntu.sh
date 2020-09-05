#!/bin/bash

DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing redis, node and python3.8
echo [$(date +"%F %T")] ================================

sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates \
    build-essential software-properties-common curl \
    autoconf libtool pkg-config gnupg-agent git ffmpeg
sudo add-apt-repository -y ppa:chris-lea/redis-server
sudo add-apt-repository -y ppa:deadsnakes/ppa
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    nodejs redis-server python3.8 \
    python3.8-dev python3-pip python3-setuptools

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing python dependencies
echo [$(date +"%F %T")] ================================

sudo python3.8 -m pip install -U pip
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
