2D Bounding Box
---------------

For tasks like object detection, we need to draw bounding boxes for the objects
of interest in the image. The interface of drawing 2D bounding boxes is shown as
follows.

.. figure:: ../media/doc/images/image_bbox_0.2.png
   :alt: 2D Bounding Boxes

   2D bounding box labeling interface

Simply click and drag on the canvas to create a bounding box. On the left side,
we show all the label categories specified in ``categories.yml`` (which you can
modify to meet your demand) and also provide options like ``Occluded`` and
``Truncated``. For traffic lights, you can also annotate their colors on the
left side.

.. figure:: ../media/doc/videos/box2d_change.gif
   :alt: Labeling bounding boxes

   Changing category/attributes of a selected box

Click on a bounding box to select the label, and press ``delete`` to delete it.
Drag the control points on the bounding box to resize it.

.. figure:: ../media/doc/videos/box2d_select-delete.gif
   :alt: Labeling bounding boxes

   Selecting, deleting and resizing bounding boxes
