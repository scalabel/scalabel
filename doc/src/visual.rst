Visualization
===================

Besides the scalabel server, we provides python scripts to Visualize the labels.

Command line
-------------
You can start the Visualization by running:

``python3 -m scalabel.vis.controller <args>``

Available arguments:

.. code-block:: bash

    --image-dir IMAGE_DIR, -i IMAGE_DIR
                            path to the image directory
    --labels LABEL_PATH, -l LABEL_PATH
                            path to the json file
    --scale SCALE, -s SCALE
                            visualization size scale
    --height HEIGHT
                            height of the image(px)
    --width WIDTH
                            width of the image(px)
    --no-attr
                            do not show attributes
    --no-box3d
                            do not show 3D bouding boxes
    --no-tags
                            do not show tags on boxes or polygons
    --no-vertices
                            do not show vertices
    --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                            output image directory with label visualization. If
                            it is set, the images will be written to the output
                            folder instead of being displayed interactively.
    --nproc NUM_PROCESS
                            number of processes for json loading and parsing


Python API
-------------

We also provides APIs for more general using cases.

Class ``scalabel.vis.viewer.LabelViewer`` is the basic class for label visualization.
It provides these methods:

.. code-block:: yaml

    - __init__():
        - ui_cfg: UIConfig
        - display_cfg: DisplayConfig
    - write():
        - out_path: str, output path
    - draw_image():
        - title: str (title of ``plt.show()``)
        - img: 3d np.array of the image
    - show_frame_attributes():
        - frame: Frame
    - draw_box2d():
        - label: Label
    - draw_box3d():
        - label: Label
        - intrinsics: Intrinsics
    - draw_poly2d():
        - labels: List[Label]

``UIConfig`` and ``DisplayConfig`` are configuration classes for specify the
LabelViewer instance