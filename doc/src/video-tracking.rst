Video Tracking
--------------

Tracking is similar to image annotation, but each object can appear in multiple
images. We can use bounding boxes or polygons to track the instances across the
frames. The labels of the same instance in the exported label file will have the
same label id.

.. figure:: ../media/doc/videos/box2d_tracking_result.gif
    :width: 600px

    An example of tracking results

2D Bounding Box Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the labeling interface of video tracking, move the slider around to move
across frames.

.. raw:: html

    <video width="600" controls autoplay loop>
        <source
            src="https://www.scalabel.ai/media/doc/videos/box2d_tracking_slider.gif.mp4"
            type="video/mp4">
        Your browser does not support the video tag.
    </video>

Bounding box interpolation
=====================================

Create a bounding box at the first frame it appears, and adjust the bounding
boxes in the subsequent frames. For a given bounding box track, the frames in
which the box is edited are considered a keyframe. The frames between keyframes
are automatically interpolated using the interpolation method selected
during project creation.

.. figure:: ../media/doc/videos/box2d_tracking_keyframe.gif
    :width: 600px


Ending object track
=====================================
For an object that disappears after certain frame, click "End Object Track" or
press ``Ctrl-E`` (``Cmd-E`` for Mac users) at the frame of its last occurrence.
The bounding box labels after this frame will be deleted.


.. figure:: ../media/doc/videos/box2d_tracking_end-track.gif
    :width: 600px


Track Linking
=====================================
Sometimes an object reappears in the frame due to occlusion or re-entrance, and
track linking enables individual tracks to be linked as a single instance.
Select a label, click ``Track-Link`` or press ``Ctrl-L`` (``Cmd-L`` for Mac
users), and click on any other tracks that you want to link with this label. The
tracks you choose to link appears in dashed lines. Click "Finish Track-Link" or
hit ``Enter`` to finish this operation.


.. figure:: ../media/doc/videos/box2d_tracking_track-link.gif
    :width: 600px

Track linking for 2D bounding box does not allow the tracks to be linked to have
overlapping frames; make sure to end object tracks correctly for all tracks
before the linking operation.


Instance Segmentation Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track labeling for instance segmentation is similar with that for bounding box.
A key difference is that for instance segmentation, overlapping frames is
allowed when linking different tracks.



Moving a segmentation label
=====================================
After labeling a segmentation label in a keyframe, adjusting each vertex in a
subsequent frame can be laborious. Press ``m`` and drag a selected label to move
the entire label around.


.. figure:: ../media/doc/videos/seg2d_tracking_move.gif
    :width: 600px



Redrawing a segmentation label
=====================================
At a different frame, sometimes it is easier to redraw the entire segmentation
label than adjusting each existing vertex. Press ``Ctrl-delete`` (``Cmd-delete``
for Mac users) to re-draw a segmentation label in the selected object track.

.. figure:: ../media/doc/videos/seg2d_tracking_redraw.gif
    :width: 600px
