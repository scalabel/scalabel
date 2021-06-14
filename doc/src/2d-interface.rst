2D Labeling Interface
----------------------

.. role::  raw-html(raw)
    :format: html

The annotation interface is shown below.

.. figure:: ../media/doc/images/annotation_interface_0.2.png
   :alt: Task Interface

   2D annotation interface

Category and attributes loaded during project creation are shown in the left
sidebar.

Jump between images by clicking the :raw-html:`<i class="fa fas fa-chevron-left
fa-xs"></i>` :raw-html:`<i class="fa fas fa-chevron-right fa-xs"></i>` buttons,
or pressing :raw-html:`&larr;`/:raw-html:`&rarr;` keys. You can also edit the
item index and hit ``Enter`` to jump to a specific image.

To zoom in/out, click the :raw-html:`<i class="fa fas fa-plus fa-xs"></i>`
:raw-html:`<i class="fa fas fa-minus fa-xs"></i>` buttons on the top-left
corner. You can also zoom by scrolling while pressing the ``Ctrl`` key
(``Cmd`` for Mac users).

.. TODO: support dragging and modify this gif.

.. figure:: ../media/doc/videos/2d_zoom-drag.gif
   :alt: Zooming and dragging the image

There are a few useful links on the top right corner of the annotation interface.
To check out the instruction page set during project creation, click the
``Instructions`` button. Toggling the keyboard shortcut window by pressing the
``Keyboard Usage`` button or the ``?`` key. Click the ``Dashboard`` button to
jump to the vendor dashboard.

To save the results of the current task, click ``Save``. Always save the task
before refreshing or leaving the annotation interface. To disable saving, turn
on the "demo mode" in the advanced options during project creation.
If auto-saving is enabled, the task will be auto-saved and the ``Save`` button
will not shown.

Once done labeling each item of the whole task, click ``Submit`` to indicate
that the whole task is finished. This action marks the task as submitted in the
project and vendor dashboards.
