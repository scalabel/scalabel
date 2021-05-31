Configuration Format
----------------------

The configuration should contain a list of category names. Additionally, the
available attributes for the dataset as well as the image resolution (if all
images have the same resolution, otherwise use the attribute "size" of each
frame) may be optionally specified. For categories where objects should be
detected, but are not tracked, set the attribute 'tracking' to false.

BDD100K example configuration in toml format:

.. code-block:: toml

    resolution = [720, 1280]

    [[attributes]]
    name = "crowd"
    toolType = "switch"
    tagText = "c"

    [[categories]]
    name = "human"
        [[categories.subcategories]]
        name = "pedestrian"

        [[categories.subcategories]]
        name = "rider"

    [[categories]]
    name = "vehicle"
       [[categories.subcategories]]
        name = "car"

       [[categories.subcategories]]
        name = "truck"

       [[categories.subcategories]]
        name = "bus"

        [[categories.subcategories]]
        name = "train"

    [[categories]]
    name = "bike"
        [[categories.subcategories]]
        name = "motorcycle"

        [[categories.subcategories]]
        name = "bicycle"

    [[categories]]
    name = "traffic light"

    [[categories]]
    name = "traffic sign"
