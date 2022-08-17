2D Video Tracking
------------------

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

.. figure:: ../media/doc/videos/box2d_tracking_slider.gif
    :width: 600px

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
For an object that disappears after certain frame, press ``Backspace`` at the
frame of its last occurrence. The bounding box labels after this frame will be
deleted.


Polygons Linking
=====================================
As in the 2D segmentaion case, sometimes an instance can be divided into
multiple parts in the image due to occlusion. To link different polygons, first
press Ctrl (Cmd) to select polygons you want to link, then press l (lowercase)
to link these selected polygons.

However, unlike the 2D segmentaion case, **not only these two polygons but also
the two entire tracks to which they belong will be merged into one**. Further
more, **unlinking is not supported**, due to the ambiguity of the assignment of
polygon components across images among new tracks. Therefore, please double
check all linking candidates before proceeding.

Track Linking
=====================================
Sometimes an object reappears in the frame due to occlusion or re-entrance, and
track linking enables individual tracks to be linked as a single instance.
Select a label, click "Link Tracks" or press ``Ctrl-L`` (``Cmd-L`` for Mac
users), and click on any other tracks that you want to link with this label. The
tracks you choose to link appears in dashed lines. Click "Finish" or
hit ``Enter`` to finish this operation.


.. figure:: ../media/doc/videos/box2d_tracking_track-link.gif
    :width: 600px

Track linking does not allow the tracks to be linked to have
overlapping frames; make sure to end object tracks correctly for all tracks
before the linking operation.


Track Breaking
=====================================
If you accidentally link two tracks that not belong to the same object. You can
click "Break track" at the frame you want to break this track. After this operation,
the original track will break into two tracks. The first one will have labels of
the original tracks from frame ``0`` to ``break_frame - 1``, the second one will
contain labels of the original track start at ``break_frame``.

.. figure:: ../media/doc/videos/box2d_tracking_track-break.gif
    :width: 600px


Single frame addition/deletion
=====================================
Under the default setting, when you draw a label at a given frame, this label will
be propagated to the end of this video. Sometimes this is unnecessary. So we provide
an operation to only draw label at the chosen frame. This could be enabled by drawing
the label while holding ``s``.

.. figure:: ../media/doc/videos/box2d_tracking_single-frame-addition.gif
   :width: 600px
   :alt: Single frame addtion

   When pressing s, the label would only be added at the given frame

Also, the default deleting operation would delete all labels after the chosen frame.
Sometimes we only want to delete labels at some specific frames. So we provide an
operation that only deletes the label at the chosen frame, while not affecting the
labels of other frames. To use it, press ``backspace`` while holding ``s``.

.. figure:: ../media/doc/videos/box2d_tracking_single-frame-deletion.gif
   :width: 600px
   :alt: Single frame deletion

   When pressing s, the label would only be deleted at the give frame


2D Instance Segmentation Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track labeling for instance segmentation is similar with that for bounding box.
