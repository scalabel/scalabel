Label Conversion
===================

Currently, we provide conversion scripts between scalabel and following formats: COCO, Waymo, KITTI, MOTChallenge, NuScenes, CrowdHuman.
They are in the ``label`` module of the  `Scalabel python package
<https://github.com/scalabel/scalabel/tree/master/scalabel>`_. We recommend
using the tools through python module invoke convention. For example,

.. code-block:: bash

    python3 -m scalabel.label.to_coco ...



from_coco
-----------------

``from_coco`` converts coco-format json files into scalabel format.
Currently, for conversion of segmentation, only ``polygon`` format is supported.

Available arguments:

.. code-block:: bash

    --input INPUT, -i INPUT
                            path to the coco-format json file
    --output OUTPUT, -o OUTPUT
                            output json path for ``det`` and ``ins_seg`` or
                            output jsons folder for ``box_track`` and ``seg_track``


to_coco
-----------------

``to_coco`` converts scalabel json files into coco format.
Now it support the conversions of four tasks: object detection as ``det``,
instance segmentation as ``ins_seg``, multi-object tracking as ``box_track`` and
multi-object tracking, or named segmentation tracking, as ``seg_track``.

To store more information in the coco-format json files, we add new
property names "videos" to the coco format. It is a list like "videos" and
"annotations", and each item has two properties: "id" and "name".

Note: For segmentation tasks, the mask conversion may not be reversible.
``Polygon`` format can be converted back with accuracy loss. Meanwhile ``RLE``
format's converting back is not supported currently, but this conversion has no loss in
mask accuracy.

Available arguments:

.. code-block:: bash

    --input INPUT, -i INPUT
                            input json path for ``det`` and ``ins_seg`` or
                            input jsons folder for ``box_track`` and ``seg_track``
    --output OUTPUT, -o OUTPUT
                            path to the output file to save the coco file
    --mode MODE, -m MODE
                            one of [det, ins_seg, box_track, seg_track]
    --mask-mode, -mm,
                            conversion mode: rle or polygon
    --nproc
                            number of processes for mot evaluation
    --config
                            config file for COCO categories


from_mot
-----------------

``from_mot`` converts MOTChallenge annotations into Scalabel format.

Available arguments:

.. code-block:: bash

    --input INPUT, -i INPUT
                            path to MOTChallenge data (images + annotations).
    --output OUTPUT, -o OUTPUT
                            Output path for Scalabel format annotations.
    --split-val SPLIT_VALIDATION
                            Split each video into train and validation parts (50 / 50).


from_waymo
-----------------

``from_waymo`` converts Waymo Open tfrecord files into Scalabel format.

Available arguments:

.. code-block:: bash

    --input INPUT, -i INPUT
                            path to MOTChallenge data (images + annotations).
    --output OUTPUT, -o OUTPUT
                            Output path for Scalabel format annotations.
    --save-images, -s SAVE
                            If the images should be extracted from .tfrecords and saved (necessary for using Waymo Open data with Scalabel format annotations).
    --use-lidar-labels USE_LIDAR
                            If the conversion script should use the LiDAR labels as GT for conversion (3D + 2D projected). Default is Camera labels (2D only).
    --nproc NPROC
                            Number of processes for conversion. Default is 4.


from_kitti
-----------------

``from_kitti`` converts KITTI annotations into Scalabel format.

Available arguments:

.. code-block:: bash

    --input-dir INPUT, -i INPUT
                            path to KITTI data (images + annotations).
    --output-dir OUTPUT, -o OUTPUT
                            Output path for Scalabel format annotations.
    --split SPLIT
                            one of [training, testing]
    --data-type DATA_TYPE
                            one of [tracking, detection]
    --nproc NPROC
                            Number of processes for conversion. Default is 4.


from_nuscenes
-----------------

``from_nuscenes`` converts NuScenes annotations into Scalabel format.

Available arguments:

.. code-block:: bash

    --input-dir INPUT, -i INPUT
                            path to NuScenes data root.
    --version VERSION, -v VERSION
                            NuScenes dataset version to convert: v1.0-trainval, v1.0-test, v1.0-mini
    --output-dir OUTPUT, -o OUTPUT
                            Output path for Scalabel format annotations.
    --splits SPLIT
                            Depending on version one of [mini_train, mini_val, train, val, test]
    --add-non-key ADD_NON_KEY
                            Add non-key frames (not annotated) to the converted data.
    --nproc NPROC
                            Number of processes for conversion. Default is 4.


to_nuscenes
-----------------

``to_nuscenes`` converts Scalabel format into a NuScenes result file.

Available arguments:

.. code-block:: bash

    --input-dir INPUT, -i INPUT
                            root directory of Scalabel label Json files or path to a label json file
    --output-dir OUTPUT, -o OUTPUT
                            path to save nuscenes formatted label file
    --mode MODE, -m MODE
                            conversion mode: detection or tracking.
    --nproc NPROC
                            Number of processes for conversion. Default is 4.
    --metadata METADATA
                            Modalities / Data used: camera, lidar, radar, map, external
