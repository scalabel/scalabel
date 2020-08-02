<p align="center"><img width=250 src="https://s3-us-west-2.amazonaws.com/scalabel-public/www/logo/scalable_dark.svg" /></p>

---

![Build & Test](https://github.com/scalabel/scalabel/workflows/Build%20&%20Test/badge.svg?branch=master)
[![Language grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/scalabel/scalabel.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/scalabel/scalabel/context:javascript)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/scalabel/scalabel.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/scalabel/scalabel/context:python)
![Docker Pulls](https://img.shields.io/docker/pulls/scalabel/www)

[Scalabel](https://www.scalabel.ai) (pronounced "scalable") is a versatile and scalable annotation platform, supporting both 2D and 3D data labeling. [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) is labeled with this tool.

![scalabel interface](https://www.scalabel.ai/doc/demo/readme/scalabel_teaser_interface.jpg)

## Demos

- [Overview video](https://go.yf.io/scalabel-video-demo)

## Try It Yourself

Below is a quick way to install dependencies and launch the Scalabel server. After launching the server, you can directly jump to [how to use the tool](#using-scalabel). If you want to know more about the installation process, please check [additional tips](#installation-tips).

**Note**: You only need to do either Step 3 or 4, but not both.

1. Check out the code

    ```shell
    git clone https://github.com/scalabel/scalabel
    cd scalabel
    ```

2. Prepare the local data directories

    ```shell
    bash scripts/setup_local_dir.sh
    ```

    If you have a local folder of images or point clouds to label, you can move them to `local-data/items`. After launching the server (finishing Step 3 or 4), the url for the images will be `localhost:8686/items`, assuming the port in the scalabel config is 8686. The url of the example image (`local-data/items/examples/cat.webp`) is [http://localhost:8686/items/examples/cat.webp](http://localhost:8686/items/examples/cat.webp). Any files in the `items` folder and subfolders will be served. Files at `local-data/items/{subpath}` are available at `{hostname}/items/{subpath}`.

3. Using Docker

    Download from dockerhub

    ```shell
    docker pull scalabel/www
    ```

    Launch the server

    ```shell
    docker run -it -v "`pwd`/local-data:/opt/scalabel/local-data" -p 8686:8686 -p 6379:6379 scalabel/www \
        node app/dist/main.js \
        --config /opt/scalabel/local-data/scalabel/config.yml \
        --max-old-space-size=8192
    ```

    Depending on your system, you may also have to increase docker's memory limit (8 GB should be sufficient).

4. Build the code yourself

    This is an alternative to using docker. We assume you have already installed [Homebrew](https://brew.sh/) if you are using Mac OS X and you have `apt-get` if you are on Ubuntu. The code requires Python 3.7 or above. Please check [how to upgrade your Python](#upgrade-python) if you don't have the right version. We use 3.8 by default. Depending your OS, run the script

    ```shell
    bash scripts/setup_osx.sh
    ```

    or

    ```shell
    bash scripts/setup_ubuntu.sh
    ```

    If you are on Ubuntu, you may need to run the script with `sudo`.

    Then you can launch the server using node:

    ```shell
    node app/dist/main.js --config ./local-data/scalabel/config.yml --max-old-space-size=8192
    ```
     Depending on your system, you may also have to increase the memory limit from 8192 (8 GB).

5. Get labels

    The collected labels can be directly downloaded from the project dashboard. The data can be follow [bdd data format](https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md). After installing the requirements and setting up the paths of the [bdd data toolkit](https://github.com/ucbdrive/bdd-data), you can visualize the labels by

    ```shell
    python3 -m bdd_data.show_labels -l <your_downloaded_label_path.json>
    ```

More installation and usage details can be find in our [documentation](http://www.scalabel.ai/doc). It also includes Windows setup.

## Using Scalabel

### Create annotation projects

The entry point of creating an project is `http://localhost:8686/create`. The page looks like

<img src="https://www.scalabel.ai/doc/demo/readme/project-creation-blank.png" width="500px">

`Project Name` is an arbitrary string. You can create multiple projects, but they cannot have duplicated names. You can choose `Item Type` and `Label Type` from the dropdown menus. An automatic page title will be provided based on the label settings. A project consists of multiple tasks. `Task Size` is the number of items (image or point cloud) in each task.

`Item List` is the list of images or point clouds to label. The format is either json or yaml with a list of frame objects in the [bdd data format](https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md). The only required field for the item list is `url`. See [examples/image_list.yml](examples/image_list.yml) for an example of image list.

`Category` and `Attributes` are the list of tags giving to each label. Typical settings are shown in [examples/categories.yml](examples/categories.yml) and [examples/bbox_attributes.yml](examples/bbox_attributes.yml). We also support multi-level categories such as [two](examples/two_level_categories.yml) and [three](examples/three_level_categories.yml) levels. Scalabel also supports [image tagging](examples/image_tags.yml).

If you want to create an annotation project to label 2d bounding boxes, the setup will looks like

<img src="https://www.scalabel.ai/doc/demo/readme/project-creation.png" width="500px">

After `ENTER` is clicked, the project will be created and two dashboards will be generated. The links to the dashboards will appear in the project creation page.

<img src="https://www.scalabel.ai/doc/demo/readme/project-creation-after-enter.png" width="500px">

`DASHBOARD` is the main dashboard for annotation progress and label downloading.

<img src="https://www.scalabel.ai/doc/demo/readme/creator-dashboard.png" width="500px">

You can download the annotation results in BDD format from the `EXPORT RESULTS` button in the toolbar on the left.

`VENDOR DASHBOARD` is for the annotation vendor to check the list of tasks.

<img src="https://www.scalabel.ai/doc/demo/readme/vendor-dashboard.png" width="500px">

The task link will lead you to each task. In our example, the task is to label 2D bounding boxes with their categories and attributes.

<img src="https://www.scalabel.ai/doc/demo/readme/bbox2d-interface.jpg">

### Semi-automatic Annotation

Because the image list is in [bdd data format](https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md), it can also contain labels within each frame. For example, you can upload an image list like [examples/image_list_with_auto_labels.json](examples/image_list_with_auto_labels.json). The labels in this email list are generated by an object detector. The labels will be automatically loaded in the tasks and shown to the annotator for adjustment.

You can use an off-shelf object detector such as [Faster RCNN](https://github.com/facebookresearch/maskrcnn-benchmark). If the results generated by the detector is in COCO
format, you can use [our script](scripts/coco2bdd.py) to convert the results to BDD format.

Another use of this function is to provide further adjustment for existing labels generated by [Scalabel](https://www.scalabel.ai). You can directly upload the exported results from a previous annotation project and the labels will show up again in the new tasks.

### Collaborative Labeling

Use the synchronization config

```shell
cp app/config/sync_config.yml local-data/scalabel/config.yml
```

Now you can open multiple sessions for the same project, and they will automatically synchronize the data.

### Other

Some to-be-released features can be enabled by launching the server with the following python script:
```shell
python3.8 -m scalabel.tools.launch_server --config ./local-data/scalabel/config.yml
```

### More Usage Info

Please go to [documentation](http://www.scalabel.ai/doc) for detailed annotation instructions and advanced usages.

## Installation Tips

### Development

We transpile or build Javascript code

```shell
npm install
node_modules/.bin/webpack --config webpack.config.js --mode=production
```

If you are debugging the code, it is helpful to build the javascript code in development mode, in which you can trace the javascript source code in your browser debugger. `--watch` tells webpack to monitor the code changes and recompile automatically.

```shell
node_modules/.bin/webpack --watch --config webpack.config.js --mode=development
```

### Upgrade Python

Our python code requires Python3.7 and above. To install the proper Python versions, we recommend [pyenv](https://github.com/pyenv/pyenv), especially for Mac users.

Homebrew on Mac can directly install `pyenv`

```shell
brew update && brew install pyenv
```

Otherwise, you can follow the `pyenv` [installation tutorial](https://github.com/pyenv/pyenv#installation). Next, install Python 3.8.2

```shell
pyenv install 3.8.2
```

Set it as global default

```shell
pyenv global 3.8.2
```

Adding the new Python to your `PATH`

```shell
export PATH=$(pyenv root)/shims:$PATH
```

On Ubuntu, you can also use [`deadsnakes` ppa](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa/+index) to install different versions of Python. Ubuntu 18.04 or later provides the package of Python 3.8 directly.

```shell
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3-pip python3-setuptools
```

### Redis security

Then, the server can be accessed at `http://localhost:8686`. You can now check out [example usage](#example-usage) to create your first annotation project. Please make sure secure your redis server following <https://redis.io/topics/security/>. By default redis will backup to local file storage, so ensure you have enough disk space or disable backups inside redis.conf.

<!-- ### Cognito Integration

Scalabel could integrate with [AWS Cognito](https://aws.amazon.com/cognito/). You can use Cognito to manage users. Once you have set up Cognito (See official tutorial [here](https://docs.aws.amazon.com/cognito/latest/developerguide/tutorials.html)), go to config file, fill the properties like below.

```yaml
userManagement: true //If set to true, then the following configs are required
cognito:
  region: "us-west-2"
  userPool: "us-west-2_tgxuoXZdf"
  clientId: "52i44u3c7fapmec4oaqto4lk121"
  userPoolBaseUri: "scalabel.auth.us-west-2.amazoncognito.com"
  callbackUri: "http://localhost:8686/callback"
``` -->

- region: Region of your cognito service.
- userPool: Pool ID - You can find it in [General Settings]
- clientID: App Client ID - You can find it in [General settings] -> [App clients] or [App integration] -> [App client settings]
- userPoolBaseUri: App Domain - You can find it in [App integration] -> [Domain name]
- callbackUri: Must exact as what you filled in [App integration] -> [App client settings]

### Backward compatibility

We are doing our best to make sure that our system can be stable and the internal data can be reused in the new updates. At the same time, we are also iterating on the internal design so that it can be more efficient and versatile. In some cases, we have to break the backward compatibility for the internal data storage. When this happens to your project, you can export the labels from the old project and import them to the new project with the new code. We definitely hope you can enjoy the new features we constantly add to Scalabel.
