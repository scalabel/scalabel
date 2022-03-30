"""
{
  "camera": {
    "extrinsic": {
      "pitch": 0.038, 
      "roll": -0.0, 
      "x": 1.7, 
      "y": 0.1, 
      "yaw": -0.0195, 
      "z": 1.22
    }, 
    "intrinsic": {
      "fx": 1590.83437, 
      "fy": 1592.79032, 
      "resx": 1440, 
      "resy": 720, 
      "u0": 771.31406, 
      "v0": 360.79945
    }
  }, 
    "instance": {
    "bbox2d": {
        "2400004": {
            "xmax": 0.36819, 
            "xmin": 0.32987, 
            "ymax": 0.65853, 
            "ymin": 0.40874, 
            "zmax": 15.57714, 
            "zmin": 14.97358
        }, 
    }
    "bbox3d": {
        "2400004": {
            "origin": [
                17.45311, 
                2.8416, 
                0.13857
            ], 
            "x": [
                -0.51131, 
                0.25348, 
                0.00786
            ], 
            "y": [
                -0.31006, 
                -0.62552, 
                0.00219
            ], 
            "z": [
                0.02309, 
                -0.00556, 
                1.68086
            ]
        }, 
    }
      "class": {
        "2400004": 24, 
        "2400037": 24, 
        "2400039": 24, 
        "2400071": 24, 
        "2400072": 24, 
        "2400086": 24, }
      "occluded": {
        "2400004": 0.0, 
        "2400037": 0.87457, 
        "2400039": 0.70672, 
        "2400071": 0.92922,}

Synscapes labels:
#       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),


    conversion = {
        0: "car",  # Car
        1: "car",  # Van
        2: "truck",  # Truck
        3: "person",  # Pedestrian
        4: "person",  # Person_sitting
        5: "rider",  # Cyclist
        6: None,  # Tram
        7: None,  # Misc
        8: None,  # DontCare
    }
"""
# The evaluation script expects one json annotation file per image with the format:
# {
#     "objects": [
#         {
#             "2d": {
#                 "modal": [xmin, ymin, w, h],
#                 "amodal": [xmin, ymin, w, h]
#             },
#             "3d": {
#                 "center": [x, y, z],
#                 "dimensions": [length, width, height],
#                 "rotation": [q1, q2, q3, q4],
#             },
#             "label": str,
#             "score": float
#         }
#     ]
# }


import numpy as np
import os
import json
from pyquaternion import Quaternion

ISO_8855 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]


def synscapes_to_cityscapes(data):
    intrinsics = data["camera"]["intrinsic"]
    extrinsics = data["camera"]["extrinsic"]
    print("extrinsics", extrinsics)
    width = intrinsics["resx"]
    height = intrinsics["resy"]

    # extrinsic_rotation = Quaternion(axis=[0, 1, 0], angle=-extrinsics["pitch"]) * Quaternion(
    #     axis=[0, 0, 1], angle=-extrinsics["yaw"]
    # )
    extrinsic_rotation = Quaternion()
    extrinsic_translation = [-extrinsics["x"], -extrinsics["y"], -extrinsics["z"]]
    extrinsic_matrix = np.array(extrinsic_rotation.transformation_matrix)
    extrinsic_matrix[:3, 3] = extrinsic_translation

    print("matrix", extrinsic_matrix[:3].tolist())
    sensor = {
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "v0": intrinsics["v0"],
        "u0": intrinsics["u0"],
        "sensor_T_ISO_8855": extrinsic_matrix[:3].tolist(),
    }

    instances = data["instance"]
    classes = instances["class"]
    class_maps = {26: "car", 27: "truck", 24: "person", 25: "rider"}

    objects = []

    for key in classes:
        label_key = classes[key]
        label = class_maps.get(label_key, None)
        if label is None:
            continue
        box2d = instances["bbox2d"][key]
        box3d = instances["bbox3d"][key]

        xmin = box2d["xmin"] * width
        xmax = box2d["xmax"] * width
        ymin = box2d["ymin"] * height
        ymax = box2d["ymax"] * height
        w2d = xmax - xmin
        h2d = ymax - ymin

        back_corner = np.array(box3d["origin"])
        x = np.array(box3d["x"])
        y = np.array(box3d["y"])
        z = np.array(box3d["z"])
        l = np.linalg.norm(x)
        w = np.linalg.norm(y)
        h = np.linalg.norm(z)
        center = back_corner + x / 2 + y / 2 + z / 2

        # Axis: x = forward, y = left, z = up
        forward = np.array([1, 0, 0])
        x_flat = np.array([x[0], x[1], 0])
        x_unit = x_flat / np.linalg.norm(x_flat)

        angle = np.arccos(np.clip(np.dot(x_unit, forward), -1.0, 1.0))
        rotation = Quaternion(axis=[0, 0, 1], angle=angle)
        qw, qx, qy, qz = rotation.elements

        objects.append(
            {
                "2d": {"amodal": [xmin, ymin, w2d, h2d]},
                "3d": {
                    "center": center.tolist(),
                    "dimensions": [l, w, h],
                    "rotation": [qw, qx, qy, qz],
                },
                "label": label,
                "score": 1.0,
            }
        )

    return {"objects": objects, "sensor": sensor, "ignore": [], "imgWidth": 1440, "imgHeight": 720}


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    meta_path = os.path.join(dir_path, "../../../local-data/items/synscapes/meta/1001.json")

    out_path = os.path.join(
        dir_path, "../../../local-data/items/cityscapes/gtBbox3d/train/aaatest/aaatest_000000_000019_gtBbox3d.json"
    )

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    get_labels = synscapes_to_cityscapes(metadata)

    with open(out_path, "w") as file:
        json.dump(get_labels, file)
