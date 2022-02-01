3D Video Tracking
---------------
Similar to 2D video tracking, 3D video tracking allows 3D bounding box labels
that are present in multiple frames. The labels of the same instance in the exported label file will have the same label id.

3D Bounding Box Tracking
~~~~~~~~~~~~~~~~~~~~~~~~
To get started, create a ``Video Tracking`` project, with the ``3D Bounding Box``
label type. 

Bounding box interpolation
==========================

Create a 3D bounding box in the first frame that an object appears, and adjust the bounding
boxes in the subsequent frames. For a given bounding box track, the frames in
which the box is edited are considered a keyframe. The frames between keyframes
are automatically interpolated using linear interpolation.

Ending object track
=====================================
For an object that disappears after certain frame, press ``Backspace`` at the
frame of its last occurrence. The bounding box labels after this frame will be
deleted.

Track linking, breaking and single frame addition/deletion
==========================================================
The functionality to link and break tracks or to create and delete labels in a single
frame are the same as in 2D video tracking. See the documentation here.
