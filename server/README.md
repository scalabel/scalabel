# Option 1: Launch backend servers with AWS
## One time actions
Locally, install the following dependencies:
````
pip install awscli --upgrade --user
pip install -U ray
````
Set up your aws credentials. Use `us-east-2` as default region. (see [this page](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-tutorial.html#tutorial-configure-cli) for details):
````
aws configure
````
Create an amazon machine image (ami):
- If available to you, use ami-02599cc60bfa86716 (us-east-2)
- Otherwise, create an EC2 instance on ubuntu 18.04, and run `./setup_sat.sh`. Add your github credentials by running `git config --global credential.helper store` then `git pull`. Then record the ami using [these instructions](https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html). Then substitute this ami for each ImageID field in compute/cluster.yaml, and in all commands below
Create an AWS security group:
````
aws ec2 create-security-group --group-name gateway-sg --description "Security group for gateway server"
````
Sample output:
````
{
    "GroupId": "sg-0e8d48691c769bba6"
}
````
- Record the `GroupId`, i.e. `sg-0e8d48691c769bba6`, for future use. Open `server/model/compute/cluster.yaml`, and on lines 61 and 76, put this `GroupId` inside the brackets for each SecurityGroupIds field, as shown below:
````
    SecurityGroupIds: [sg-0e8d48691c769bba6]
````
Give the group permissions to itself and your public IP address, which you can look up online:
````
aws ec2 authorize-security-group-ingress --group-name gateway-sg --protocol all --source-group gateway-sg
aws ec2 authorize-security-group-ingress --group-name gateway-sg --protocol all --cidr <PUBLIC_IP>/32
````
To later revoke access (if your IP changed, for example):
````
aws ec2 revoke-security-group-ingress --group-name gateway-sg --protocol all --cidr <OLD_PUBLIC_IP>/32
````
Create a key for the gateway server
````
cd server/model/gate
aws ec2 create-key-pair --key-name gateway-key --query "KeyMaterial" --output text > gateway-key.pem
chmod 400 gateway-key.pem
````
## Actions to launch servers
Make a new terminal tab (the ray tab) to launch the ray cluster:
````
cd server/model/compute
ray up cluster.yaml
````
In the original tab (the gateway tab), launch the gateway server and record its IP (the following commands also store it in a bash variable). Make sure to insert your `GroupId` in the first command:
````
instanceID=$(aws ec2 run-instances --image-id ami-02599cc60bfa86716  --count 1 --instance-type m5.large --key-name gateway-key --security-group-ids <GroupId> --query "Instances[0].InstanceId")
instanceID=${instanceID%\"}
instanceID=${instanceID#\"}
instanceIP=$(aws ec2 describe-instances --instance-ids $instanceID --query "Reservations[0].Instances[0].PublicIpAddress")
instanceIP=${instanceIP%\"}
instanceIP=${instanceIP#\"}
echo $instanceIP
````
In a new tab (sat tab), update config files with the correct IPs:
`cd sat`
- In `app/config/default_config.yml`, set modelGateHost to the gateway IP (the `instanceIP` that was displayed after running the above commands)
- In `server/model/gate/gate_config.yml`, set machineHost to the ray public DNS (you can find this under the ray head instance description on the AWS console)
In the gateway tab, update the gateway server with these changes:
````
ssh -t -i gateway-key.pem ubuntu@$instanceIP 'sudo rm -rf sat && git config --global credential.helper store && git clone -b model_server_scaling --single-branch https://github.com/ucbdrive/sat.git && source ~/.profile && cd sat/server/model && protoc -I proto/ proto/model-server.proto --go_out=plugins=grpc:proto && python3 -m grpc_tools.protoc -I proto/ --python_out=compute/ --grpc_python_out=compute/ proto/model-server.proto'
````
Still in the gateway tab, copy the gate config to the EC2 instance:
````
scp -i gateway-key.pem gate_config.yml ubuntu@$instanceIP:~/sat/server/model/gate/gate_config.yml
````
In the ray tab, start running the cluster:
````
ray exec cluster.yaml 'cd sat/server/model/compute && python3 model_server.py'
````
In the gateway tab, start running the server:
````
ssh -t -i gateway-key.pem ubuntu@$instanceIP 'cd sat/server/model/gate && go build -o gateway . && ./gateway --config gate_config.yml'
````
In the sat tab, run the following instruction to launch the http server:
````
cd sat
go build -i -o ./bin/scalabel ./server/http
npm install package.json
node_modules/.bin/npx webpack --config webpack.config.js --mode=production
mkdir data
cp app/config/default_config.yml data/config.yml
./bin/scalabel --config ./data/config.yml
````
Access the speed test page at:
``localhost:8686/dev/speed_test.html``.
Make sure to copy the config every time the IP changes:
````
cp app/config/default_config.yml data/config.yml
````
## Actions to shut down servers
In the ray tab, kill ray server
````
ray down cluster.yaml
````
In the gateway tab, kill gateway server
````
aws ec2 terminate-instances --instance-ids $instanceID
````
# Option 2: Launch backend locally
Follow the installation instructions in `setup_backend.sh` (make sure to use the Mac installation instructions if necessary).
Set host IPs to local host:
- In `app/config/default_config.yml`, set modelGateHost to 127.0.0.1
- In `server/model/gate/gate_config.yml`, set machineHost to 127.0.0.1
Compile the proto buffers:
````
cd server/model
protoc -I proto/ proto/model-server.proto --go_out=plugins=grpc:proto
python -m grpc_tools.protoc -I proto/ --python_out=compute/ --grpc_python_out=compute/ proto/model-server.proto
````
Compile and start the gateway server:
````
cd gate
go build -o gateway .
./gateway --config gate_config.yml
````
Start the python server:
````
cd ../compute
python model_server_local.py --local
````
Follow the instructions [here](https://github.com/ucbdrive/sat) to launch the http server and go to ``localhost:8686/dev/speed_test.html``.
