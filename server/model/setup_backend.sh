#Installation
#Install packages
sudo apt-get update
sudo apt-get install autoconf automake libtool curl make g++ unzip python3-pip

#Set up go for Ubunut
#For Mac install, see https://golang.org/doc/install#install
echo 'export GOROOT=/usr/local/go' >> ~/.profile
echo 'GOPATH=$HOME/go' >> ~/.profile
echo 'PATH=$PATH:$GOPATH/bin' >> ~/.profile
echo 'alias python=python3' >> ~/.profile
source ~/.profile
curl -O https://dl.google.com/go/go1.10.4.linux-amd64.tar.gz
tar xvf go1.10.4.linux-amd64.tar.gz
sudo chown -R root:root ./go
sudo mv go /usr/local

#Install go packages
sudo go get github.com/gorilla/websocket github.com/aws/aws-sdk-go github.com/mitchellh/mapstructure github.com/satori/go.uuid gopkg.in/yaml.v2
sudo go get -u google.golang.org/grpc github.com/golang/protobuf/protoc-gen-go

#Install python packages
pip3 install -U ray
pip3 install setproctitle psutil jupyter protobuf grpcio-tools grpcio

#Install repo
git clone https://github.com/ucbdrive/sat.git
cd sat

#Install packages for grpc for Ubuntu
#For Mac install, see http://google.github.io/proto-lens/installing-protoc.html
# and https://grpc.io/docs/quickstart/go.html#go-version
curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google

#Compilation
#Compile proto files
cd server/model
protoc -I proto/ proto/model-server.proto --go_out=plugins=grpc:proto
python -m grpc_tools.protoc -I proto/ --python_out=compute/ --grpc_python_out=compute/ proto/model-server.proto

#General SAT compilation (see https://github.com/ucbdrive/sat)
cd ../..
sudo go build -i -o ./bin/scalabel ./server/http
sudo apt-get install npm
npm install
node_modules/.bin/npx webpack --config webpack.config.js --mode=development
mkdir data
cp app/config/default_config.yml data/config.yml
