Python Tools
===================

We provide some useful scripts to handle scalabel projects and I/O. They are in
the ``tools`` module of the `Scalabel python package
<https://github.com/scalabel/scalabel/tree/master/scalabel>`_. We recommend
using the tools through python module invoke convention. For example,

.. code-block:: bash

    python3 -m scalabel.tools.prepare_data ...


prepare_data
-------------------

``prepare_data`` converts videos or images to a data folder and an image list
that can be directly used for creating Scalabel projects. Assume all the images
are in ``.jpg`` format.

Available arguments:

.. code-block:: bash

    --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                            path to the video/images to be processed
    --input-list INPUT_LIST [INPUT_LIST ...]
                            List of input directories and videos for processing.
                            Each line in each file is a file path.
    --out-dir OUT_DIR, -o OUT_DIR
                            output folder to save the frames
    --fps FPS, -f FPS     the target frame rate. default is the original video
                            framerate.
    --scratch             ignore non-empty folder.
    --s3 S3               Specify S3 bucket path in bucket-name/subfolder
    --url-root URL_ROOT   Url root used as prefix in the yaml file. Ignore it if
                            you are using s3.
    --start-time START_TIME
                            The starting time of extracting frames in seconds.
    --max-frames MAX_FRAMES
                            Max number of output frames for each video.
    --no-list             do not generate the image list.
    --jobs JOBS, -j JOBS  Process multiple videos in parallel.

You can download our testing video at
https://scalabel-public.s3-us-west-2.amazonaws.com/demo/dancing.mp4. The video
was from `Youtube <https://www.youtube.com/watch?v=-ZTsgbrdoI8>`_ with Creative
Commons Attribution license.

Convert video to a folder of images and generate the image list:

.. code-block:: bash

    python3 -m scalabel.tools.prepare_data --start-time 5 --max-frames 100 \
        -i dancing.mp4 -o ./dancing --fps 3 \
        --url-root http://localhost:8686/items/dancing1k


The above command also starts extracting the frames from the 5th second and
stops at 100th frame. The extraction fps is 3 and it is assumed that the images
are served from the `items` dir. The command will also generate
``image_list.yml`` in the output folder. It will be ready to be used in the
Scalabel project creation. You can put multiple videos after ``-i`` to process
those videos togther and put all the images in the ``image_list.yml``

If you want to upload the images to the s3 and generate the corresponding image
list, you can do

.. code-block:: bash

    python3 -m scalabel.tools.prepare_data --start-time 5 --max-frames 100 \
        -i ~/Downloads/dancing.mp4  --fps 3 -o dancing-s3 \
        --s3 scalabel-public/demo/dancing

Please refer to the `s3 doc
<http://boto3.readthedocs.io/en/latest/guide/s3-example-creating-buckets.html>`_
to configure your S3 bucket.


We also support multiple inputs, as well as image folder. The command below will
read the images from the folder `dancing-s3` and copy them to the output folder
before breaking down the video. The script will think we will create a frame
list of two videos. The images in the generated list will be assigned to one of
two video names.

.. code-block:: bash

    python3 -m scalabel.tools.prepare_data --start-time 5 --max-frames 100  \
        --fps 3 -i ./dancing-s3 dancing.mp4  -o test

If you want to process a folder of videos together with a list of videos, you
can use

.. code-block:: bash

    python3.8 -m scalabel.tools.prepare_data -i videos/*.mov  --input-list \
        videos.txt -o frames_all/



edit_labels
-------------------

``edit_labels`` provides utilities to edit image and label list.

Add url prefix to the name field in the frames and assign it to the url field

.. code-block:: bash

    python3 -m scalabel.tools.edit_labels --add-url http://localhost:8686/items -i \
        input.json -o output.json
