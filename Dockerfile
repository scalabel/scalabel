FROM ubuntu:18.04
EXPOSE 8686
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    npm \
    nodejs \
    curl \
    git &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scalabel
RUN chmod -R a+w /opt/scalabel

COPY package*.json ./
RUN npm install

COPY scripts ./scripts

COPY . .
RUN ./node_modules/.bin/npx webpack --config webpack.config.js --mode=production; \
    rm -f app/dist/tsconfig.tsbuildinfo
