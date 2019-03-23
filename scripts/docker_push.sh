#!/usr/bin/env bash

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker build . -t scalabel/www
if [[ $? -ne 0 ]]; then
    echo "[!] docker build failed" >&2
    exit 1
fi
docker push scalabel/www
