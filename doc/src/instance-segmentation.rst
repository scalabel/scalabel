Instance Segmentation
---------------------

Annotating instance segmentation involves drawing polygons. Simply click on the
image to start a label, and close the path to finish drawing. Double-click on a
label to select it.

Dragging the midpoint on an edge turns the midpoint into a vertex. Click the
midpoint of an edge while pressing ``c`` makes the edge a bezier curve. You can
adjust the control points to get tight-fitting curves. To revert back to a
normal edge, click on a control point while pressing ``c``.

.. figure:: ../media/doc/videos/seg2d_draw.gif

   Drawing a segmentation label

Sometimes an instance can be divided into multiple parts in the image due to
occlusion. To link different polygons, select a segmentation label, and press
``Ctrl-L`` (``Cmd-L`` for Mac users) or the ``Link`` button to start linking.
Click all labels that you want to link, and hit ``Enter`` to finish linking.

.. figure:: ../media/doc/videos/seg2d_link.gif

   Linking segmentation labels

Segmentation labels often share borders with each other. To make segmentation
annotation more convenient, Scalabel supports vertex and edge sharing. When
drawing a segmentation label, adding a new vertex at the position of an existing
vertex shares the reference of the two vertices. The edge between two vertices
that are both shared by two segmentation labels will also be shared. When
adjusting a vertex or an edge with shared reference, all segmentation labels
involved will be changed accordingly.

.. Quick Draw is a useful tool for border sharing. When drawing a segmentation label that needs to share a border with an existing label,
.. press ``Ctrl-D`` (``Cmd-D`` for Mac users) or the ``Quick Draw`` button to start Quick Draw mode. First select a polygon to share the
.. border with, and then select the starting vertex and the ending vertex of the shared border. Press ``Alt`` to toggle between two
.. possible shared paths. Hit ``Enter`` to end Quick Draw.

.. .. figure:: ../media/doc/videos/seg2d_quickdraw.gif

..     Quick Draw

.. To delete a single vertex, click on the vertex while pressing ``d``. When drawing a segmentation label in progress, pressing ``d``
.. deletes the last vertex drawn on the image.

Lane Marking
~~~~~~~~~~~~

Lane marking is similar to segmentation labeling, except that the path is not
closed. Hit ``Enter`` to finish drawing a label.

.. figure:: ../media/doc/videos/lane.gif

    Lane marking

Drivable Area
~~~~~~~~~~~~~

Similarly as other instance segmentation tasks, you can annotate the
drivable area in the image. Details about the drivable area can be found
in this `blog <http://bair.berkeley.edu/blog/2018/05/30/bdd/>`_.

