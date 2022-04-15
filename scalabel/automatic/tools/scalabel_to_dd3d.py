import numpy as np
import cv2
import os
import json
from pyquaternion import Quaternion
from scalabel.automatic.model_repo.dd3d.structures.boxes3d import GenericBoxes3D
from scalabel.automatic.model_repo.dd3d.structures.pose import Pose

from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import draw_boxes3d_cam
from scalabel.automatic.model_repo.dd3d.utils.kitti_metadata import metadata as kitti_metadata


def scalabel_to_dd3d(labels):
    boxes3d = []
    for label in labels:
        if label["box3d"] is None:
            continue
        location = label["box3d"]["location"]
        dimension = label["box3d"]["dimension"]
        orientation = label["box3d"]["orientation"]
        rotation = orientation[1] + np.pi / 2
        # rotation = 0

        x, y, z = location
        w, h, l = dimension

        box_pose = Pose(
            wxyz=Quaternion(axis=[1, 0, 0], radians=np.pi / 2) * Quaternion(axis=[0, 0, 1], radians=-rotation),
            tvec=np.float64([x, y, z]),
        )
        box3d = GenericBoxes3D(box_pose.quat.elements, box_pose.tvec, [w, l, h])
        boxes3d.append(box3d)
    return boxes3d


if __name__ == "__main__":
    sample_id = 1080
    dir_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(dir_path, "../../..", f"local-data/items/synscapes-sample/img/rgb/{sample_id}.png")
    filename = "synscapes-1080_export_2022-04-12_15-36-09.json"
    file_path = os.path.join(dir_path, "../../..", f"local-data/annotations/{filename}")
    with open(file_path) as f:
        file_data = json.load(f)

    labels = file_data["frames"][0]["labels"]

    boxes = scalabel_to_dd3d(labels)
    vectorized = np.array([box.vectorize().cpu().numpy() for box in boxes]).squeeze()
    classes = [0 for _ in boxes]
    intrinsics = np.array([[1590.83437, 0.0000, 771.31406], [0.0000, 1592.79032, 360.79945], [0.0000, 0.0000, 1.0000]])
    image_cv2 = cv2.imread(img_path)
    vis_image = draw_boxes3d_cam(image_cv2, vectorized, classes, kitti_metadata, intrinsics)

    outpath = "test.png"
    cv2.imwrite(outpath, vis_image)
