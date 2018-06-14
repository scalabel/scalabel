FROM golang
EXPOSE 8686
ADD . .
RUN echo "---\nport: 8686\ndataDir: \"./data\"\nprojectPath: \".\"\n..."  >> config.yml
RUN go get gopkg.in/yaml.v2
RUN go build -i -o bin/sat ./server/go
CMD ./bin/sat --config ./config.yml
