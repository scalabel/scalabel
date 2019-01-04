###Launch gateway server:
Follow instructions here to install gRPC for go and Protocol Buffers:
````
 https://grpc.io/docs/quickstart/go.html#go-version
````
Install the following dependencies:
````
go get github.com/gorilla/websocket
````
Generate the gRPC client interface:
```
cd server/model
protoc -I proto/ proto/model-server.proto --go_out=plugins=grpc:proto
python -m grpc_tools.protoc -I proto/ --python_out=compute/ --grpc_python_out=compute/ proto/model-server.proto
```
Compile and start to gateway server:
````
cd gate
go build -o gateway .
./gateway --config gate_config.yml
````
Start the python server:
````
cd ../compute
python model_server.py
````
Follow the instructions [here](https://github.com/ucbdrive/sat) to launch the http server and go to ``localhost:8686/dev/speed_test.html``.
