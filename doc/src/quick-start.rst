.. _quick-start:

Quick Start
------------------------------------------


The entry point of creating an project is http://localhost:8686/create. The page
looks like

.. image:: ../media/doc/images/project-creation-blank.png
  :width: 500

``Project Name`` is an arbitrary string. You can create multiple projects, but
they cannot have duplicated names.

``Item Type`` and ``Label Type`` could be choosen from the dropdown menus to define
what to annotate in the project. An automatic page title will be provided based
on the label settings.

There are two ways to construct a labeling project.

1. Provide config with separate files.

    ``Item List`` is the list of images or point clouds to label. The format is
    either json or yaml with a list of frame objects in the :ref:`Scalabel format`.
    The only required field for the item list is ``url``. See
    `examples/image_list.yml
    <https://github.com/scalabel/scalabel/blob/master/examples/image_list.yml>`_
    for an example of image list.

    ``Category`` and ``Attributes`` are the list of tags giving to each label.
    Typical settings are shown in `examples/categories.yml
    <https://github.com/scalabel/scalabel/blob/master/examples/categories.yml>`_ and
    `examples/bbox_attributes.yml
    <https://github.com/scalabel/scalabel/blob/master/examples/bbox_attributes.yml>`_.
    We also support multi-level categories such as `two
    <https://github.com/scalabel/scalabel/blob/master/examples/two_level_categories.yml>`_
    and `three
    <https://github.com/scalabel/scalabel/blob/master/examples/three_level_categories.yml>`_
    levels. Scalabel also supports `image tagging
    <https://github.com/scalabel/scalabel/blob/master/examples/image_tags.yml>`_.

    ``Item List`` and ``Category`` must be provided.

2. Provide config with a single file.

    Click ``Submit single file`` to submit a single integrated config file.
    An example config file is given in `examples/dataset.json
    <https://github.com/scalabel/scalabel/blob/master/examples/dataset.json>`_
    following :ref:`Scalabel format`.

A project could be divided into multiple tasks. ``Task Size`` is the number
of items (image or point cloud) in each task.

If you want to create an annotation project to label 2d bounding boxes, the
setup will look like

.. image:: ../media/doc/images/project-creation.png
  :width: 500

After ``SUBMIT`` is clicked, the project will be created and two dashboards will
be generated. The links to the dashboards will appear in the project creation
page.

.. image:: ../media/doc/images/project-creation-after-enter.png
  :width: 500

``DASHBOARD`` is the main dashboard for annotation progress and label
downloading.

.. image:: ../media/doc/images/creator-dashboard.png
  :width: 500

You can download the annotation results in :ref:`Scalabel Format` from the
``DOWNLOAD LABELS`` button in the toolbar on the left.

``VENDOR DASHBOARD`` is for the annotation vendor to check the list of tasks.

.. image:: ../media/doc/images/vendor-dashboard.png
  :width: 500

The task link will lead you to each task. In our example, the task is to label
2D bounding boxes with their categories and attributes.

.. image:: ../media/doc/images/bbox2d-interface.png
  :width: 500
