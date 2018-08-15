<p align="center"><img width=250 src="https://s3-us-west-2.amazonaws.com/scalabel-public/www/logo/scalable_dark.svg" /></p>

--------------------------------------------------------------------------------


[![Build Status](https://travis-ci.com/ucbdrive/scalabel.svg?token=9QKS6inVmkjyhrWUHjqT&branch=master)](https://travis-ci.com/ucbdrive/scalabel)

Scalabel (pronounced "scalable") is a versatile and scalable tool that supports all kinds of annotations needed in a driving database. It supports bounding box, semantic instance segmentation, and video tracking.

## Setup ##
1. Checkout the code
    ```
    git clone git@github.com:ucbdrive/scalabel.git
    cd scalabel
    ```
2. Create a directory <data_dir> to store the server data:
    ```
    mkdir ../data
    ```
    Or, for Windows users:
    ```
    mkdir ..\data
    ```
3. Launch server. There are two options, either (i) to build
 with Docker or (ii) to build by yourself.
    1. Build and run a Docker image from the Dockerfile.

        Build by yourself
        ```
        docker build . -t 'scalabel/server'
        ```
        Or
        ```
        docker pull scalabel/server
        ```
        After getting the docker image, you can run the server
        ```
        docker run -it -v `pwd`/../data:/data -p 8686:8686 scalabel/server
        ```
    2. Build the server by yourself.
        1. Install GoLang. Refer to the [instruction page](https://golang.org/doc/install) for details.
        2. Install GoLang dependency
        ```
        go get -u gopkg.in/yaml.v2 github.com/aws/aws-sdk-go github.com/mitchellh/mapstructure
        ```
        
        3. Compile the server 
        ```
        go build -i -o $GOPATH/bin/scalabel ./server/go
        ```
        For Windows users, use
        ```
        go build -i -o %GOPATH%\bin\scalabel.exe .\server\go
        ```
        
         Note that you may choose your own path for the server executable. We
        use $GOPATH/bin/scalabel as it conforms to golang best practice, 
        but if your GOPATH is not configured, this will not work.
        
        4. Specify basic configurations (e.g. the port to start the server, 
        the data directory to store the image annotations, etc) in your own 
        `config.yml`. Refer to `app/config/default_config.yml` for the default configurations.
        You can choose the storage method by specifying database in config:
        setting database to "file" to store data in json format locally, "dynamodb" to store data in aws. 
        5. Launch the server by running 
        ```
        $GOPATH/bin/scalabel --config app/config/default_config.yml
        ```
        For Windows users, run:
        ```
        %GOPATH%\bin\scalabel.exe --config app\config\default_config.yml
        ```
        
         If you used a different server path when compiling, make sure to use
        the correct path here.
    
3. Access the server through the specifed port (we use `8686` as the default port
specified in the `config.yml`)
    ```
    http://localhost:8686
    ```

# Usage and Demo

Please check the [documentation](doc/usage.md) for detailed usage instructions.
