FROM ubuntu:18.04
EXPOSE 8686
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    npm \
    nodejs \
    curl \
    git &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -o go.tgz -O https://dl.google.com/go/go1.11.linux-amd64.tar.gz; \
    tar -C /usr/local -xzf go.tgz; \
    rm go.tgz; \
    export PATH="/usr/local/go/bin:$PATH";
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

WORKDIR /opt/scalabel
RUN chmod -R a+w /opt/scalabel

COPY package*.json ./
RUN npm install

COPY . .
RUN bash scripts/install_go_packages.sh
RUN ./node_modules/.bin/npx webpack --config webpack.config.js --mode=production && \
    rm -rf node_modules
RUN go build -i -o bin/scalabel ./server/http
