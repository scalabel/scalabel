#!/usr/bin/env bash

cd scalabel/bot
git clone https://github.com/fidler-lab/polyrnn-pp.git
mv polyrnn-pp polyrnn_pp
python3.6 -m pip install -U -r polyrnn_pp/requirements.txt
python3.6 -m pip install -U -r requirements.txt
cd polyrnn_pp
./models/download_and_unpack.sh 
cd ../../..
export PYTHONPATH="${PYTHONPATH}:scalabel/bot/polyrnn_pp/src"