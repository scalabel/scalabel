#!/bin/bash

ROOT=1
DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
PARENT_DIR=$(dirname ${DIR})
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
    echo "    ${SCRIPT_NAME} [-u] args ..."
    echo ""
    title "DESCRIPTION"
    echo "    Scalabel setup script for Ubuntu"
    echo ""
    title "OPTIONS"
    echo "    -h, --help                    Print this help"
    echo "    -u, --user                    Run as user"
    echo ""
}

while getopts "hu" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        u) # run as user
            ROOT=0;;
    esac
done

if [[ $ROOT -eq 0 ]]
then
    echo ""
    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] RUNNING AS USER $(whoami)
    echo [$(date +"%F %T")] ================================
    echo ""

    if ! command -v nvm >/dev/null
    then
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] Installing NVM
        echo [$(date +"%F %T")] ================================
        echo ""

        curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash

        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
    else
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] NVM already installed
        echo [$(date +"%F %T")] ================================
    fi

    NVM_VER=$(nvm --version)
    echo ""
    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] NVM version: ${NVM_VER}
    echo [$(date +"%F %T")] ================================
    echo ""
    nvm install stable

    if ! command -v redis-server >/dev/null
    then
        echo ""
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] Installing Redis
        echo [$(date +"%F %T")] ================================
        echo ""

        wget https://download.redis.io/redis-stable.tar.gz
        tar -xzvf redis-stable.tar.gz
        cd redis-stable
        make -j$(nproc)
        # make test -j$(nproc)

        cd ..
        rm -rf redis-stable.tar.gz

        REDIS_PATH="${PARENT_DIR}/redis-stable/src/redis-server"
        shopt -s expand_aliases
        echo "alias redis-server=\"${REDIS_PATH}\"" >> ~/.bash_aliases
        source ~/.bash_aliases
    else
        echo ""
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] Redis already installed
        echo [$(date +"%F %T")] ================================
    fi

    REDIS_VER=$(redis-server --version | awk '{print substr($3, 3)}')
    echo ""
    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] Redis version: ${REDIS_VER}
    echo [$(date +"%F %T")] ================================
    echo ""

    if ! command -v conda >/dev/null
    then
        echo [$(date +"%F %T")] ================================
        echo [$(date +"%F %T")] Installing conda
        echo [$(date +"%F %T")] ================================

        . ${DIR}/install_conda.sh ${PARENT_DIR}

        eval "$(${PARENT_DIR}/conda/bin/conda shell.bash hook)"
        echo '[[ -f ${PARENT_DIR}/conda/bin/conda ]] && eval "$(${PARENT_DIR}/conda/bin/conda shell.bash hook)"' >> ${HOME}/.bashrc
        echo ""
    fi

    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] Creating conda environment
    echo [$(date +"%F %T")] ================================

    conda env create -f ${DIR}/env.yml

else
    echo ""
    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] RUNNING AS ROOT
    echo [$(date +"%F %T")] ================================
    echo ""

    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] Installing redis, node and python3.8
    echo [$(date +"%F %T")] ================================
    echo ""

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

    echo ""
    echo [$(date +"%F %T")] ================================
    echo [$(date +"%F %T")] Installing python dependencies
    echo [$(date +"%F %T")] ================================
    echo ""

    sudo python3.8 -m pip install -U pip
    python3.8 -m pip install --user -U --ignore-installed -r ${DIR}/requirements.txt
fi

echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Setting up local directories
echo [$(date +"%F %T")] ================================
echo ""

. scripts/setup_local_dir.sh

echo ""
echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Installing node packages
echo [$(date +"%F %T")] ================================

npm install --save-dev --max_old_space_size=8000

echo ""
echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Compiling source code
echo [$(date +"%F %T")] ================================
echo ""

node --max_old_space_size=8000 node_modules/.bin/webpack --config webpack.config.js --mode=production

echo ""
echo [$(date +"%F %T")] ================================
echo [$(date +"%F %T")] Post-installation actions
echo [$(date +"%F %T")] ================================

title "Set up packages"
echo "python setup.py develop"
echo ""
echo ""
