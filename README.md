<p align="center"><img width=250 src="https://s3-us-west-2.amazonaws.com/scalabel-public/www/logo/scalable_dark.svg" /></p>

---

![Build & Test](https://github.com/scalabel/scalabel/workflows/Build%20&%20Test/badge.svg?branch=master)
[![Language grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/scalabel/scalabel.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/scalabel/scalabel/context:javascript)
[![Language grade:
Python](https://img.shields.io/lgtm/grade/python/g/scalabel/scalabel.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/scalabel/scalabel/context:python)
![Docker Pulls](https://img.shields.io/docker/pulls/scalabel/www)

[Scalabel](https://www.scalabel.ai) (pronounced "scalable") is a versatile and scalable annotation platform, supporting both 2D and 3D data labeling. [BDD100K](https://www.bdd100k.com/) is labeled with this tool.

# Install http
1. Init dirs
    ```bash
    bash scripts/setup_local_dir.sh
    ```
2. Run docker-compose
    ```bash
    docker-compose up -d
    ```
# Install https
1. Скопируйте и сконфигурируйте.conf
    ```bash
    cp conf.d/nginx.conf-example conf.d/nginx.conf
    vim .conf
    ```
Нужно написать `autoindex off;` чтобы не показывать список файлов

2. Настройте server_name, ssl_certificate, ssl_certificate_key 
3. Run docker-compose
    ```bash
    docker-compose up -d
    ```


[**Documentation**](https://doc.scalabel.ai/) |
[**Overview Video**](https://go.yf.io/scalabel-video-demo) |
[**Discussion**](https://groups.google.com/g/scalabel) |
[**Contributors**](https://github.com/scalabel/scalabel/graphs/contributors)

![scalabel interface](https://doc.scalabel.ai/_images/box2d_tracking_result.gif)
