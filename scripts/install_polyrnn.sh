#!/usr/bin/env bash

cd scalabel/bot
git clone https://bitbucket.org/datu-ai/experimental.git
cd experimental/fast-seg-label
mv polyrnn orig_code
patch -p0 -i ../../polyrnn.patch
mv orig_code polyrnn
cd ../..
mkdir experimental_models
cd experimental_models
aws s3 cp s3://datu-public-data/models . --recursive
cd ../../..
export PYTHONPATH="${PYTHONPATH}:scalabel/bot/experimental/fast-seg-label/polyrnn"