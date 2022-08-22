2D Instance Segmentation
--------------------------

Annotating instance segmentation involves drawing polygons. Simply click on the
image to start a label, and close the path to finish drawing. Click on the
boundary to select the label.

Dragging the midpoint on an edge turns the midpoint into a vertex. Click the
midpoint of an edge while pressing ``c`` makes the edge a bezier curve. You can
adjust the control points to get tight-fitting curves. To revert back to a
normal edge, click on a control point while pressing ``c``.

.. figure:: ../media/doc/videos/seg2d_draw.gif

   Drawing a segmentation label

Sometimes an instance can be divided into multiple parts in the image due to
occlusion. To link different polygons, first press ``Ctrl`` (``Cmd``) to select
polygons you want to link, then press ``l`` (lowercase) to link these selected
polygons. Press ``L`` (uppercase) to unlink the label.

.. figure:: ../media/doc/videos/seg2d_link.gif

   Linking and unlinking segmentation labels

We also support vertex deletion. To delete a vertex from a drawn polygon, press
``d`` first, and then click on the control point you want to delete. The second
way to use vertex deletion is when drawing a new polygon. When the drawing is
not finished, press ``d`` could delete previous drawn vertex.

.. figure:: ../media/doc/videos/seg2d_vertex_deletion.gif

   Deleting vertex

Segmentation labels often share borders with each other. To make segmentation
annotation more convenient, Scalabel supports vertex and edge sharing. When
drawing a segmentation label, adding a new vertex at the position of an existing
vertex shares the reference of the two vertices. The edge between two vertices
that are both shared by two segmentation labels will also be shared. When
adjusting a vertex or an edge with shared reference, all segmentation labels
involved will be changed accordingly.

Quick Draw is a useful tool for border sharing. When drawing a segmentation
label that needs to share a border with an existing label, press ``Ctrl-D``
(``Cmd-D`` for Mac users) or the ``Quick Draw`` button to start Quick Draw mode.
First select a polygon to share the border with, and then select the starting
vertex and the ending vertex of the shared border. Press ``Alt`` to toggle
between two possible shared paths. Hit ``Enter`` to end Quick Draw.

.. figure:: ../media/doc/videos/seg2d_quickdraw.gif

    Quick Draw

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

