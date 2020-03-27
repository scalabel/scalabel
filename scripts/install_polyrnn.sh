#!/usr/bin/env bash

cd scalabel/bot/models
# download from bitbucket
git clone https://github.com/fidler-lab/polyrnn-pp.git
# download models separately?
# apply the patch
cd polyrnn-pp
patch -p0 -i ../polyrnn.patch
cd ../../../..
# add this to bashrc?
export PYTHONPATH="${PYTHONPATH}:scalabel/bot/models/polyrnn-pp-pytorch-small/code"