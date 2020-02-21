FROM ubuntu:18.04
EXPOSE 8686

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    gnupg-agent \
    software-properties-common \
    curl \
    git \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    redis-server

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash

RUN apt-get update && apt-get install -y --no-install-recommends nodejs

WORKDIR /opt/scalabel
RUN chmod -R a+w /opt/scalabel

COPY . .

RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -r scripts/requirements.txt

RUN npm install -g npm@latest && npm install --max_old_space_size=8000


RUN ./node_modules/.bin/npx webpack --config webpack.config.js --mode=production; \
    rm -f app/dist/tsconfig.tsbuildinfo
