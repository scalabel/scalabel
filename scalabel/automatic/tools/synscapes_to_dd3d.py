import json
import numpy as np
from pyquaternion import Quaternion
import os

from sklearn.metrics import coverage_error

from scalabel.automatic.model_repo.dd3d.structures.boxes3d import GenericBoxes3D
from scalabel.automatic.model_repo.dd3d.structures.pose import Pose
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti

from detectron2.data import DatasetCatalog, MetadataCatalog

from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti
from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import draw_boxes3d_cam
from scalabel.automatic.model_repo.dd3d.utils.visualization import float_to_uint8_color


def map_axes(arr):
    return np.array([-arr[1], -arr[2], arr[0]])


def synscapes_to_dd3d(metadata):
    boxes = []
    for key in metadata["instance"]["bbox3d"]:
        # if metadata["instance"]["class"][key] != 26:
        #     continue
        # Synscapes axes: x = forward, y = left, z = up
        # Kitti axes: x = right, y = down, z = forward
        resx = 1440
        resy = 720
        box3d = metadata["instance"]["bbox3d"][key]
        box2d = metadata["instance"]["bbox2d"][key]
        xmin = box2d["xmin"] * resx
        xmax = box2d["xmax"] * resx
        ymin = box2d["ymin"] * resy
        ymax = box2d["ymax"] * resy

        origin = map_axes(box3d["origin"]) - map_axes([1.7, 0.1, 1.22])
        forward = map_axes(box3d["x"])
        left = map_axes(box3d["y"])
        up = map_axes(box3d["z"])

        length = np.linalg.norm(forward)
        width = np.linalg.norm(left)
        height = np.linalg.norm(up)

        # Origin is bottom right back corner. To get center of box, we add
        # half of forward, left, and up vectors
        center = origin + forward / 2 + left / 2 + up / 2

        # Get rotation about y axis
        box_forward_unit = forward / np.linalg.norm(forward)
        ego_right_unit = np.array([1, 0, 0])
        rot_y = np.arccos(np.clip(np.dot(box_forward_unit, ego_right_unit), -1.0, 1.0))
        if box_forward_unit[2] < 0:
            rot_y = np.pi * 2 - rot_y

        box_pose = Pose(
            wxyz=Quaternion(axis=[1, 0, 0], radians=np.pi / 2) * Quaternion(axis=[0, 0, 1], radians=rot_y),
            # wxyz=Quaternion(axis=[0, 0, 1], radians=rot_y),
            tvec=np.float64(center),
        )

        box3d_dd3d = GenericBoxes3D(box_pose.quat.elements, box_pose.tvec, [width, length, height])
        boxes.append({"box3d": box3d_dd3d, "box2d": [xmin, xmax, ymin, ymax]})

    return boxes


if __name__ == "__main__":
    # # Draw ground truth boxes for 100 samples
    # for sample_id in range(1, 100):
    #     dir_path = os.path.dirname(os.path.abspath(__file__))
    #     img_path = os.path.join(dir_path, "../../..", f"local-data/items/synscapes-sample/img/rgb/{sample_id}.png")
    #     labels_path = os.path.join(dir_path, "../../..", f"local-data/items/synscapes-sample/meta/{sample_id}.json")

    #     with open(labels_path) as f:
    #         metadata = json.load(f)

    #     boxes = synscapes_to_dd3d(metadata)
    #     boxes3d = [b["box3d"] for b in boxes]

    #     metadata = MetadataCatalog.get("kitti_3d")
    #     metadata.thing_classes = VALID_CLASS_NAMES
    #     metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]
    #     metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}

    sample_id = 6
    dir_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(dir_path, "../../..", f"local-data/items/synscapes-sample/img/rgb/{sample_id}.png")
    labels_path = os.path.join(dir_path, "../../..", f"local-data/items/synscapes-sample/meta/{sample_id}.json")

    # [[1590.83437, 0.0000, 771.31406, 0.0], [0.0000, 1592.79032, 360.79945, 0.0], [0.0000, 0.0000, 1.0000, 0.0]]
    # 1590.83437 0.0000 771.31406 0.0 0.0000 1592.79032 360.79945 0.0 0.000 0.0000 1.0000 0.0
    meta_lines = [
        "P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
        "P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
        "P2: 1590.83437 0.0000 771.31406 0.0 0.0000 1592.79032 360.79945 0.0 0.000 0.0000 1.0000 0.0",
        "P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03",
        "R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01",
        "Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01",
        "Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01",
    ]
    with open(labels_path) as f:
        metadata = json.load(f)

    meta_outpath = os.path.join(
        "/Users/elrich/code/eth/simple_kitti_visualization", "examples/kitti/calib", f"{sample_id}.txt"
    )
    labels_outpath = os.path.join(
        "/Users/elrich/code/eth/simple_kitti_visualization", "examples/kitti/label_2", f"{sample_id}.txt"
    )

    boxes = synscapes_to_dd3d(metadata)

    with open(meta_outpath, "w") as f:
        for line in meta_lines:
            f.write(line + "\n")

    with open(labels_outpath, "w") as f:
        for box in boxes:
            w, l, h, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box["box3d"])
            xmin, xmax, ymin, ymax = box["box2d"]

            line = f"Car 0.00 0 {alpha} {xmin} {ymin} {xmax} {ymax} {h} {w} {l} {x} {y} {z} {rot_y}"

            print("box", line)
            f.write(line + "\n")
