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

In the visualization window, you may use these keys for controlling:

.. code-block:: yaml

    - n / p: Show next or previous image
    - Space: Start / stop animation
    - t: Toggle 2D / 3D bounding box (if avaliable)
    - a: Toggle the display of the attribute tags on boxes or polygons.
    - c: Toggle the display of polygon vertices.
    - Up: Increase the size of polygon vertices.
    - Down: Decrease the size of polygon vertices.


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
        - img: 3d np.array of the image
        - title: Optional[str] (title of ``plt.show()``)
    - draw_attributes(): Draw frame attributes
        - frame: Frame
    - draw_box2d():
        - labels: List[Label]
    - draw_box3d():
        - labels: List[Label]
        - intrinsics: Intrinsics
    - draw_poly2d():
        - labels: List[Label]
        - alpha: int (0.5 as default)

``UIConfig`` and ``DisplayConfig`` are configuration classes for specify the
LabelViewer instance.

Below is an simple example to use LabelViewer:

.. code-block:: python

    from scalabel.vis.viewer import DisplayConfig, LabelViewer, UIConfig

    # img: np.ndarray
    # labels: List[Label]

    viewer = LabelViewer()
    viewer.draw_image(img)
    viewer.draw_box2d(labels)
    viewer.show()

For advanded usage, you may refer the implementation of ``scalabel.vis.controller.ViewController``
as an example, or check the source code of ``LabelViewer``.
