3D Bounding Box
---------------

As with point clouds, we allow creating 3D bouding boxes on images.

Creating a Bounding Box
~~~~~~~~~~~~~~~~~~~~~~~
* Default creation: Press ``Space`` to create a new bounding box with size 1x1x1 at position ``(0, 0, 10)`` (10m in front of the camera).
After creation the box will be automatically selected and can be adjusted.

* Spanning mode: Click the ``Activate Span`` button to activate spannings mode. In this mode
you can define a custom bounding box by clicking on the pane to generate four points. These points will be automatically connected by lines and define a cuboidal skeleton of the bounding box. Once the skeleton has been generated, press ``Space`` to convert it to a bounding box. Press ``Enter`` to finish the creation.

Note that spanning mode requires a ground plane label to be present in the image.
See below for how to edit the image ground plane.

Editing Ground Plane
~~~~~~~~~~~~~~~~~~~~
For spanning mode, a ground plane label is used to determine the 3d location of 
click coordinates. A flat ground plane label is created for each image by default, 
with center located at ``(0, 1.5, 10)``, i.e. 10m in front of the camera, 1.5m down.

To toggle whether the ground plane is shown on the image, press ``g``.

For most images the ground plane may need to be rotated to match the ground orientation.
This can be done by selecting the plane with a double click and using the rotation 
handles to rotate the plane. A ground plane that matches the ground orientation 
should result in higher 3d Bounding Box accuracy.

Top View Homography
~~~~~~~~~~~~~~~~~~~
To assist with 3D labeling on images, a top-view homography is available. This
can be opened to the side or below the current view by opening a split view and
selecting ``HOMOGRAPHY``. As with spanning mode, the top view homography requires
that the image has a ground plane label. 
