#!/usr/env bash

if ! [ -x "$(command -v npm)" ]; then
    echo "install npm first"
    exit 1
fi

if ! [ -x "$(command -v eslint)" ]; then
    npm install -g eslint
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PC_PATH=$DIR/../.git/hooks/pre-commit
if [ -f $PC_PATH ]; then
   rm $PC_PATH
fi
ln -s $DIR/pre-commit.sh $DIR/../.git/hooks/pre-commit
chmod +x $DIR/pre-commit.sh
