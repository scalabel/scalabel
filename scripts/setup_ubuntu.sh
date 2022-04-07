#!/bin/bash

ROOT=0
DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
LINE=$(printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' |tr ' ' '-')

# Display underlined title to improve readability of script output
function title()
{
    echo
    echo "$@"
    echo "${LINE}"
}

function Help()
{
    # Display Help
    title "SYNOPSIS"
    echo "    ${SCRIPT_NAME} [-h] args ..."
    echo ""
    title "DESCRIPTION"
    echo "    Scalabel setup script for Ubuntu"
    echo ""
    title "OPTIONS"
    echo "    -h, --help                    Print this help"
    echo "    -r, --root                    Run as root"
    echo ""
}

while getopts ":h" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        r) # run as root
            ROOT=1;;
    esac
done

if [[ $ROOT -eq 0 ]]; then
    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] RUNNING AS USER $(whoami)
    echo [$(date +"%F %T")] ================================

    if [[ ! hash nvm 2>/dev/null ]]; then
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] Installing NVM
        echo [$(date +"%F %T")] ================================

        curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash
        nvm --version
    fi

    nvm install stable && nvm use stable

    if [[ ! hash redis-server 2>/dev/null ]]; then
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] Installing Redis
        echo [$(date +"%F %T")] ================================

        wget https://download.redis.io/redis-stable.tar.gz
        tar -xzvf redis-stable.tar.gz
        cd redis-stable
        make -j$(nproc)

        REDIS_VER=$(src/redis-server --version | awk '{print substr($3, 3)}')
        echo "Installed redis version: ${REDIS_VER}"
        cd ..
        rm -rf redis-stable.tar.gz

        REDIS_PATH="${DIR}/redis/redis-${REDIS_VER}/src/redis-server"
        shopt -s expand_aliases
        echo "alias redis-server=\"${REDIS_PATH}\"" >> ~/.bash_aliases
        source ~/.bash_aliases
    fi

    if [[ ! hash conda 2>/dev/null ]]; then
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] Installing conda
        echo [$(date +"%F %T")] ================================

        . ${DIR}/scripts/install_conda.sh
    fi

    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] Creating conda environment
    echo [$(date +"%F %T")] ================================

    conda env create -f ${DIR}/scripts/env.yml

else
    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] RUNNING AS ROOT
    echo [$(date +"%F %T")] ================================

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
        python3.8-dev python3-pip python3-setuptools python3.8-venv

    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] Installing python dependencies
    echo [$(date +"%F %T")] ================================

    sudo python3.8 -m pip install -U pip
    python3.8 -m pip install --user -U --ignore-installed -r ${DIR}/scripts/requirements.txt
fi

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing node packages
echo [$(date +"%F %T")] ================================

npm install --max_old_space_size=8000

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Compiling source code
echo [$(date +"%F %T")] ================================

node --max_old_space_size=8000 node_modules/.bin/webpack --config webpack.config.js --mode=production

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Post-installation actions
echo [$(date +"%F %T")] ================================

title "Set up packages"
echo "python setup.py develop"
title "Set up local directories"
echo "scripts/setup_local_dir.sh"
echo ""
echo ""
