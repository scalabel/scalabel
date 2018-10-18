<p align="center"><img width=250 src="https://s3-us-west-2.amazonaws.com/scalabel-public/www/logo/scalable_dark.svg" /></p>

--------------------------------------------------------------------------------


[![Build Status](https://travis-ci.com/ucbdrive/scalabel.svg?token=9QKS6inVmkjyhrWUHjqT&branch=master)](https://travis-ci.com/ucbdrive/scalabel)
[![Language grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/ucbdrive/scalabel.svg)](https://lgtm.com/projects/g/ucbdrive/scalabel/context:javascript)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ucbdrive/scalabel.svg)](https://lgtm.com/projects/g/ucbdrive/scalabel/context:python) 

[Scalabel](https://www.scalabel.ai) (pronounced "scalable") is a versatile and scalable tool that supports various kinds of annotations needed for training computer vision models, especially for driving environment. [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) is labeled with this tool.

![scalabel interface](https://s3-us-west-2.amazonaws.com/www.scalabel.ai/images/scalabel_teaser_interface.jpg)

## Demos

- [Overview video](https://go.yf.io/scalabel-video-demo)
- [2D Bounding box](https://go.yf.io/scalabel-demo-box2d)
- [2D Segmentation](https://go.yf.io/scalabel-demo-seg2d)
- [2D Bounding box tracking](https://go.yf.io/scalabel-demo-box2d-tracking)
- [2D Segmentation tracking](https://go.yf.io/scalabel-demo-seg2d-tracking)
- [2D Lane marking](https://go.yf.io/scalabel-demo-lane)
- [3D Bounding box](https://go.yf.io/scalabel-demo-box3d)
- [3D Bounding box tracking](https://go.yf.io/scalabel-demo-box3d-tracking)

## Try It Yourself

More installation and usage details can be find in our [documentation](http://www.scalabel.ai/doc). It also includes Windows setup.

1. Check out the code

    ```
    git clone git@github.com:ucbdrive/scalabel.git
    cd scalabel
    ```

2. Compile the code

    There are two alternative ways to get the compiled code
    
    1. Usage docker (recommended if you only need to run the code)
    
        Download from dockerhub
        ```
        docker pull scalabel/www
        ```
        
        or build the docker image yourself
        
        ```
        docker build . -t scalabel/www
        ```
    
    2. Compile the code yourself (recommended if you want to customize the source code)
        
        Install [golang](https://golang.org/doc/install), [nodejs and npm](https://nodejs.org/en/download/).
        
        Compile Go server code
        ```
        go get github.com/aws/aws-sdk-go github.com/mitchellh/mapstructure \ 
            gopkg.in/yaml.v2 github.com/satori/go.uuid
        go build -i -o ./bin/scalabel ./server/http
        ```
        
        Transpile or build Javascript code
        ```
        npm install
        node_modules/.bin/npx webpack --config webpack.config.js --mode=production
        ```
        
        **Note** If you are debugging the code, it is helpful to build the javascript code in development mode, in which you can trace the javascript source code in your browser debugger.
        ```
        node_modules/.bin/npx webpack --config webpack.config.js --mode=development
        ```

3. Prepare data directory

    ```
    mkdir data
    cp app/config/default_config.yml data/config.yml
    ```
    
4. Launch the server

    If using docker,
    ``` 
    docker run -it -v `pwd`/data:/opt/scalabel/data -p 8686:8686 scalabel/www \
        /opt/scalabel/bin/scalabel --config /opt/scalabel/data/config.yml
    ```
    
    Otherwise
    ``` 
    ./bin/scalabel --config ./data/config.yml
    ```
    
    Then, the server can be accessed at `http://localhost:8686`. You can now check out [example usage](#example-usage) to create your first annotation project.
    
5. Get labels
    
    The collected labels can be directly downloaded from the project dashboard. The data can be follow [bdd data format](https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md). After installing the requirements and setting up the paths of the [bdd data toolkit](https://github.com/ucbdrive/bdd-data), you can visualize the labels by
    ``` 
    python3 -m bdd_data.show_labels.py -l <your_downloaded_label_path.json>
    ```
    
## Usage

### Create annotation projects

The entry point of creating an project is `http://localhost:8686/create`. The page looks like

<img src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/project_creation_page.png" width="500px">


`Project Name` is an arbitrary string. You can create multiple projects, but they cannot have duplicated names. You can choose `Item Type` and `Label Type` from the dropdown menus. An automatic page title will be provided based on the label settings. A project consists of multiple tasks. `Task Size` is the number of items (image or point cloud) in each task. 

`Item List` is the list of images or point clouds to label. The format is either json or yaml with a list of frame objects in the [bdd data format](https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md). The only required field for the item list is `url`. See [examples/image_list.yml](examples/image_list.yml) for an example of image list. 

`Category` and `Attributes` are the list of tags giving to each label. Typical settings are shown in [examples/categories.yml](examples/categories.yml) and [examples/bbox_attributes.yml](examples/bbox_attributes.yml). We also support multi-level categories such as [two](examples/two_level_categories.yml) and [three](examples/three_level_categories.yml) levels. Scalabel also supports [image tagging](examples/image_tags.yml).

If you want to create an annotation project to label 2d bounding boxes, the setup will looks like

<img src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/project_creation_bbox2d.png" width="500px">

### More Usage Info

Please go to [documentation](http://www.scalabel.ai/doc) for detailed annotation instructions and advanced usages.
