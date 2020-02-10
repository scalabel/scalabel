FROM ubuntu:18.04
EXPOSE 8686

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    npm \
    nodejs \
    curl \
    git \
    python3.8 \
    python3-pip \
    python3-setuptools \
    redis-server && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scalabel
RUN chmod -R a+w /opt/scalabel

COPY . .

RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -r scripts/requirements.txt

RUN npm install -g npm@latest && npm install


RUN ./node_modules/.bin/npx webpack --config webpack.config.js --mode=production; \
    rm -f app/dist/tsconfig.tsbuildinfo
