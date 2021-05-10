Configuration Format
----------------------

The configuration should contain a list of category names. Additional
attributes like ignore mapping (i.e. categories that belong to a certain
class, but should be ignored), name_mapping (category names that should be
mapped to a different one), and image resolution (if all images have the
same resolution, otherwise use the attribute "size" of each frame) may be
optionally specified. For categories where objects should be detected, but
are not tracked, set the attribute 'tracking' to false.

Note that each category can contain two optional attributes: "id" for
converting between COCO and Scalabel format with given category ids and
supercategory for defining meta-classes that join semantically similar
classes during evaluation.

BDD100K example configuration in toml format:

.. code-block:: toml

    resolution = [720, 1280]

    [name_mapping]
    bike = "bicycle"
    caravan = "car"
    motor = "motorcycle"
    person = "pedestrian"
    van = "car"

    [ignore_mapping]
    "other person" = "pedestrian"
    "other vehicle" = "car"
    "trailer" = "truck"

    [[categories]]
    supercategory = "human"
    id = 1
    name = "pedestrian"
    tracking = true

    [[categories]]
    supercategory = "human"
    id = 2
    name = "rider"
    tracking = true

    [[categories]]
    supercategory = "vehicle"
    id = 3
    name = "car"
    tracking = true

    [[categories]]
    supercategory = "vehicle"
    id = 4
    name = "truck"
    tracking = true

    [[categories]]
    supercategory = "vehicle"
    id = 5
    name = "bus"
    tracking = true

    [[categories]]
    supercategory = "vehicle"
    id = 6
    name = "train"
    tracking = true

    [[categories]]
    supercategory = "bike"
    id = 7
    name = "motorcycle"
    tracking = true

    [[categories]]
    supercategory = "bike"
    id = 8
    name = "bicycle"
    tracking = true

    [[categories]]
    supercategory = "traffic light"
    id = 9
    name = "traffic light"
    tracking = false

    [[categories]]
    supercategory = "traffic sign"
    id = 10
    name = "traffic sign"
    tracking = false

