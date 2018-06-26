FROM golang
EXPOSE 8686
ADD . .
RUN go get gopkg.in/yaml.v2
RUN go build -i -o bin/sat ./server/go
CMD ./bin/sat --config ./app/config/default_config.yml
