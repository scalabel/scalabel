Label Conversion
===================

Currently, we provide conversion scripts between scalabel and coco formats.
They are in the ``label`` moduel of the  `Scalabel python package
<https://github.com/scalabel/scalabel/tree/master/scalabel>`_. We recommend
using the tools through python module invoke convention. For example,

.. code-block:: bash

    python3 -m scalabel.label.to_coco ...

To allow for more information store in the coco-format json files, we add a new
property names "videos" to the coco format. It is a list like "videos" and
"annotations", and each item is a have two properties: "id" and "name".

from_coco
-----------------

``from_coco`` converts coco-format json files into scalabel format.
Currently, for conversion of segmentation, only ``polygon`` format is supported.

Available arguments:

.. code-block:: bash

    --label INPUT, -i INPUT
                            path to the coco-format json file
    --out-dir OUT_DIR, -o OUT_DIR
                            output json path for ``det`` and ``ins_seg`` or
                            output jsons folder for ``box_track`` and ``seg_track``

to_coco
-----------------

``to_coco`` converts scalabel json files into coco format.
Now it support the conversions of four tasks: object detection as ``det``,
instance segmentation as ``ins_seg``, multi-object tracking as ``box_track`` and
multi-object trackings, or named segmentation tracking, as ``seg_track``.

Note that, for segmentation tasks, the mask conversion may not be reversible.
``Polygon`` format can be converted back with accuracy loss. Meanwhile ``RLE``
format's converting back is not supported currently, but this conversion has no loss in
mask accuracy.

Available arguments:

.. code-block:: bash

    --label INPUT, -l INPUT
                            path to the video/images to be processed
    --output OUT_DIR, -o OUT_DIR
                            output folder to save the frames
    --height HEIGHT
                            height of images
    --width WIDTH
                            width of images
    --mode MODE, -m MODE
                            one of [det, ins_seg, box_track, seg_track]
    --remove-ignore, -ri
                            remove the ignored annotations from the label file
    --ignore-as-class, -ic
                            put the ignored annotations to the `ignored` category
    --mask-mod, -mm,
                            conversion mode: rle or polygon
    --nproc
                            number of processes for mot evaluation
    --config
                            config file for COCO categories