FROM ubuntu:20.04
EXPOSE 8686

ENV DEBIAN_FRONTEND=noninteractive

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
    libopenmpi-dev \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-setuptools

# Latest redis source
RUN add-apt-repository ppa:chris-lea/redis-server

# nodejs 12
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash

RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs redis-server

WORKDIR /opt/scalabel
RUN chmod -R a+w /opt/scalabel

COPY . .

RUN python3.9 -m pip install -U pip && \
    python3.9 -m pip install -r scripts/requirements.txt

RUN python3.9 setup.py install

RUN npm install -g npm@latest && npm ci --max_old_space_size=8000


RUN ./node_modules/.bin/webpack --config webpack.config.js --mode=production; \
    rm -f app/dist/tsconfig.tsbuildinfo
