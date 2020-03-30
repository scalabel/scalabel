#!/usr/bin/env bash

cd scalabel/bot
git clone https://bitbucket.org/datu-ai/experimental.git
mkdir experimental_models
cd experimental_models
aws s3 cp s3://datu-s3-drive/polyrnn . --recursive
cd ../../..
export PYTHONPATH="${PYTHONPATH}:scalabel/bot/experimental/fast-seg-label/polyrnn_scalabel"