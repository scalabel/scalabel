Hybrid
------
Allow multiple sensors.

Point Cloud Projects
~~~~~~~~~~~~~~~~~~~~
For point cloud projects you can include image data, which can
be opened in a split view. If the calibration between the sensors is provided as 
extrinsic parameters, the image data will be transformed to align with the
point cloud labels. See the project below as example.

Image projects 
~~~~~~~~~~~~~~
For image projects you can provide point cloud data. If sensor extrinsic parameters
are provided, the point cloud data will be aligned with the image labels. 
Point cloud data can be shown as an overlay on top of the image, or in a side view.

Point cloud color scheme
========================
If point cloud data is provided as supporting data in a image project, the
point cloud can be colored with the image pixel information. See below for an example. 
