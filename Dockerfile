FROM ubuntu:18.04
EXPOSE 8686
RUN apt-get update && apt-get install -y npm nodejs wget git

RUN wget https://dl.google.com/go/go1.11.linux-amd64.tar.gz -O go.tgz; \
    tar -C /usr/local -xzf go.tgz; \
    rm go.tgz; \
    export PATH="/usr/local/go/bin:$PATH";
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN go get github.com/aws/aws-sdk-go github.com/mitchellh/mapstructure \
    gopkg.in/yaml.v2 github.com/satori/go.uuid

WORKDIR /opt/scalabel
RUN chmod -R a+w /opt/scalabel

COPY package*.json ./
RUN npm install

COPY . .
RUN ./node_modules/.bin/npx webpack --config webpack.config.js --mode=production
RUN go build -i -o bin/scalabel ./server/go