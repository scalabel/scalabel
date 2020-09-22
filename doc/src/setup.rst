Installation
~~~~~~~~~~~~~~

You can either set up the tool with the provided docker image or build the tool
by yourself. Once you have the tool set up, you can open http://localhost:8686
to create your annotation project. Below is a quick way to install dependencies
and launch the Scalabel server. After launching the server, you can directly
jump to :ref:`quick-start`.


**Note**: You only need to do either Step 3 or 4, but not both.

1.  Check out the code

    .. code-block:: bash

        git clone https://github.com/scalabel/scalabel
        cd scalabel

2.  Prepare the local data directories

    .. code-block:: bash

        bash scripts/setup_local_dir.sh

    If you have a local folder of images or point clouds to label, you can move
    them to ``local-data/items``. After launching the server (finishing Step 3
    or 4), the url for the images will be http://localhost:8686/items, assuming
    the port in the scalabel config is 8686. The url of the example image
    ``local-data/items/examples/cat.webp`` is
    http://localhost:8686/items/examples/cat.webp. Any files in the `items`
    folder and subfolders will be served. Files at
    ``local-data/items/{subpath}`` are available at
    ``{hostname}/items/{subpath}``.

3.  Using Docker

    Download from dockerhub

    .. code-block:: bash
    
        docker pull scalabel/www

    Launch the server

    .. code-block:: bash

        docker run -it -v "`pwd`/local-data:/opt/scalabel/local-data" -p \
            8686:8686 -p 6379:6379 scalabel/www \
            node app/dist/main.js \
            --config /opt/scalabel/local-data/scalabel/config.yml \
            --max-old-space-size=8192


    Depending on your system, you may also have to increase docker's memory
    limit (8 GB should be sufficient).

4.  Build the code yourself

    This is an alternative to using docker. We assume you have already installed
    `Homebrew <https://brew.sh/>`_ if you are using Mac OS X and you have
    ``apt-get`` if you are on Ubuntu. The code requires Python 3.7 or above.
    Please check [how to upgrade your Python](#upgrade-python) if you don't have
    the right version. We use 3.8 by default. Depending your OS, run the script

    .. code-block:: bash
    
        bash scripts/setup_osx.sh


    or

    .. code-block:: bash

        bash scripts/setup_ubuntu.sh


    If you are on Ubuntu, you may need to run the script with `sudo`.

    Then you can launch the server using node:

    .. code-block:: bash

        node app/dist/main.js --config ./local-data/scalabel/config.yml \
            --max-old-space-size=8192
    
    Depending on your system, you may also have to increase the memory limit
    from 8192 (8 GB).

5.  Get labels

    The collected labels can be directly downloaded from the project dashboard.
    The labels follow
    :ref:`Scalabel Format`.
    After installing the requirements and setting up the paths of the
    `BDD100K toolkit <https://github.com/bdd100k/bdd100k>`_,
    you can visualize the labels by

    .. code-block:: bash
    
        python3 -m bdd_data.vis.labels -l <your_downloaded_label_path.json>
    
