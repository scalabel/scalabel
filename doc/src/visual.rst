Visualization
===================

Besides the scalabel server, we provides python scripts to Visualize the labels.

Command line
-------------
You can start the Visualization by running:

``python3 -m scalabel.vis.label <args>``

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
    --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                            output image directory with label visualization. If
                            it is set, the images will be written to the output
                            folder instead of being displayed interactively.
    --range-begin RANGE_BEGIN
                            from which frame to visualize
    --range-end RANGE_END
                            up to which frame to visualize
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

Class ``scalabel.vis.label.LabelViewer`` is the basic class for label visualization.
It provides these methods:

.. code-block:: yaml

    - __init__():
        - ui_cfg: UIConfig (UIConfig as default)
    - draw():
        - image: 3d np.array of the image
        - frame: Frame
    - show():
    - save():
        - out_path: str, output path
    - draw():
        - image: 3d np.array of the image
        - frame: Frame
        - with_attr: bool (True as default)
        - with_box2d: bool (True as default)
        - with_box3d: bool (False as default)
        - with_poly2d: bool (True as default)
        - with_ctrl_points: bool (False as default)
        - with_tags: bool (True as default)
        - ctrl_point_size: float (2.0 as default)
    - draw_image():
        - img: 3d np.array of the image
        - title: Optional[str] (title of ``plt.show()``)
    - draw_attributes(): Draw frame attributes
        - frame: Frame
    - draw_box2d():
        - labels: List[Label]
        - with_tags: bool (True as default)
    - draw_box3d():
        - labels: List[Label]
        - intrinsics: Intrinsics
        - with_tags: bool (True as default)
    - draw_poly2d():
        - labels: List[Label]
        - alpha: int (0.5 as default)
        - with_tags: bool (True as default)
        - with_ctrl_points: bool (False as default)
        - ctrl_point_size: float (2.0 as default)

``UIConfig`` is configuration classes for specify the LabelViewer instance.

Below is an simple example to use LabelViewer:

.. code-block:: python

    from scalabel.vis.label import DisplayConfig, LabelViewer, UIConfig

    # img: np.ndarray
    # labels: List[Label]

    viewer = LabelViewer()
    viewer.draw(img, frame)
    viewer.show()

For advanded usage, you may refer the implementation of ``scalabel.vis.controller.ViewController``
as an example, or check the source code of ``LabelViewer``.
