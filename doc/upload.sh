#!/usr/bin/env bash

make html
aws s3 sync _build/src/html/ s3://www.scalabel.ai/doc --exclude ".DS_Store" --acl public-read
