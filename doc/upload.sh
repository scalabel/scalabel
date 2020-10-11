#!/usr/bin/env bash

make html
aws s3 sync _build/src/html/ s3://doc.scalabel.ai/ --exclude ".DS_Store" --acl public-read
