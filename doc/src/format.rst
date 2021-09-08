Scalabel Format
--------------------------

Scalabel format defines the protocol for importing image lists with optional
:ref:`automatic labels<Auto Label>` and exporting the manual annotations. This
is also the format for `BDD100K dataset
<https://www.bdd100k.com>`_.

Schema of the format is shown below. You can also use our `Typescript
<https://github.com/scalabel/scalabel/blob/master/app/src/types/export.ts>`_
and Python type definitions. Most of the fields are optional depending
on your purpose. If you only want to upload a list of images when creating a
project, you only need ``url``. ``videoName`` is used to group frames for each
tracking task. If you are annotating bounding boxes, you can ignore `poly2d` and
other label types.

``Item List``, ``Categories``, ``Attributes`` can be uploaded with separate
files. Or they could be contained in a single file, following the exporting format.

Exporting Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exporting format has the following fields.

.. code-block:: yaml

    - frames [ ]:
        - item001
        - item002
        ...
    - config:
        - image_size: (optional, valid when all images have the same size)
            - width: int
            - height: int
        - attributes [ ]:
            - name: string
            - toolType: string (can be 'switch' or 'list')
            - tagText: string (acronym when showing)
            - values: string[]
            - tagPrefix: string
        - categories: string[]

Each item in the ``frame`` field is an image with several fields.
``attributes``, ``categories`` are the list of tags given
to each label in images. Fields of item are given below.

.. code-block:: yaml
    
    - name: string
    - url: string
    - videoName: string (optional)
    - attributes: a dictionary of frame attributes
    - intrinsics
        - focal: [x, y]
        - center: [x, y]
        - nearClip:
    - extrinsics
        - location
        - rotation
    - timestamp: int64 (epoch time ms)
    - frameIndex: int (optional, frame index in this video)
    - size:
        - width: int
        - height: int
    - labels [ ]:
        - id: string
        - index: int
        - category: string (classification)
        - manualShape: boolean
        - manualAttributes: boolean
        - score: float
        - attributes: a dictionary of label attributes
        - box2d:
            - x1: float
            - y1: float
            - x2: float
            - y2: float
        - box3d:
            - alpha:
            - orientation: 
            - location: ()
            - dimension: (3D point, height, width, length)
        - poly2d:
            - vertices: [][]float (list of 2-tuples [x, y])
            - types: string
            - closed: boolean
        - graph: (optional)
            - nodes [ ]:
                - location: [x, y] or [x, y, z]
                - category: string
                - visibility: string (optional)
                - type: string (optional)
                - id: string (optional)
            - edges [ ]:
                - source: string
                - target: string
                - type: string (optional)
            - type: string (optional)


More details about the fields

* labels

    * index: index of the label in an image or a video
    * manualShape: whether the shape of the label is created or modified manually
    * manualAttributes: whether the attribute of the label is created or
      modified manually
    * score: the confidence or some other ways of measuring the quality of the label.
    * box3d - follows the convention in the KITTI dataset.
        * alpha: observation angle if there is a 2D view
        * location: 3D center of the box, stored as 3D point in camera coordinates, meaning the axes (x,y,z) point right, down, and forward.
        * orientation: 3D orientation of the bounding box, stored as axis angles in the same coordinate frame as the location.
        * dimension: 3D box size, with length in x direction, height in y direction and width in z direction
    
    * poly2d

        * types: Each character corresponds to the type of the vertex with the 
          same index in vertices. ‘L’ for vertex and ‘C’ for control point of a
          bezier curve.
        * closed: true for polygon and otherwise for path

    * graph

        * nodes
            * location: 2D or 3D coordinates.
            * category: Either joint name or type of segmentation (see closed in `poly2d`).
            * visibility: Visibility of joint for pose.
            * type: Type of vertex for segmentation (see type in `poly2d`).
            * id: Unique ID.

        * edges
            * source: Source node of the edge.
            * target: Target node of the edge.
            * type: Type of edge.

        * type: Specification of graph.