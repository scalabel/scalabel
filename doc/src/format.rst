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
    - size: [width, height]
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


More details about the fields

* labels

    * index: index of the label in an image or a video
    * manualShape: whether the shape of the label is created or modified manually
    * manualAttributes: whether the attribute of the label is created or
      modified manually
    * score: the confidence or some other ways of measuring the quality of the label.
    * box3d

        * alpha: observation angle if there is a 2D view
        * orientation: 3D orientation of the bounding box, used for 3D point
          cloud annotation
        * locatoin: 3D point, x, y, z, center of the box
        * dimension: 3D box size
    
    * poly2d

        * types: Each character corresponds to the type of the vertex with the 
          same index in vertices. ‘L’ for vertex and ‘C’ for control point of a
          bezier curve.
        * closed: true for polygon and otherwise for path