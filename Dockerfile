FROM ubuntu:18.04
EXPOSE 8686
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    npm \
    curl \
    git &&\
    rm -rf /var/lib/apt/lists/*
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash
RUN apt-get install -y nodejs

WORKDIR /opt/scalabel
RUN chmod -R a+w /opt/scalabel

COPY package*.json ./
RUN npm install -g npm@latest
RUN npm install

COPY scripts ./scripts

COPY . .
RUN ./node_modules/.bin/npx webpack --config webpack.config.js --mode=production; \
    rm -f app/dist/tsconfig.tsbuildinfo
