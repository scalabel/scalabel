import numpy as np
import cv2
import os
import json
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from scalabel.automatic.model_repo.dd3d.structures.boxes3d import GenericBoxes3D
from scalabel.automatic.model_repo.dd3d.structures.pose import Pose

from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import draw_boxes3d_cam
from scalabel.automatic.model_repo.dd3d.utils.kitti_metadata import metadata as kitti_metadata

test_names = [
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151268662404.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151428162404.jpg",
]

# idxs = [77, 81]


def nuscenes_to_dd3d(dataroot, version):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    nusc.list_categories()
    results = []
    for scene in [nusc.scene[81]]:
        print("scene", scene["token"], scene["description"])
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sensor = "CAM_FRONT"
            cam_front_data = nusc.get("sample_data", sample["data"][sensor])
            cs_record = nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
            ego_pose_token = cam_front_data["ego_pose_token"]
            ego_pose = nusc.get("ego_pose", ego_pose_token)
            ego_pose = {
                "timestamp": ego_pose["timestamp"],
                "rotation": ego_pose["rotation"],
                "translation": ego_pose["translation"],
            }
            intrinsics = cs_record["camera_intrinsic"]
            filename, annotations, K = nusc.get_sample_data(cam_front_data["token"])
            boxes = []
            for ann in annotations:
                if "human" not in ann.name and "vehicle" not in ann.name:
                    continue
                bbox3d = GenericBoxes3D(ann.orientation, ann.center, ann.wlh)
                boxes.append(bbox3d)
            results.append((filename, boxes, intrinsics, ego_pose))
            sample_token = sample["next"]

    return results


if __name__ == "__main__":
    dataroot = "local-data/items/nuscenes-set-1"
    version = "v1.0-trainval"
    results = nuscenes_to_dd3d(dataroot, version)
    poses = {"poses": [pose for _, _, _, pose in results]}
    print("poses length", len(poses["poses"]))
    pose_outfile = "nuscenes-poses-81.json"
    with open(pose_outfile, "w") as f:
        json.dump(poses, f)

    # for (filename, boxes, intrinsics) in results:
    #     if not os.path.exists(filename):
    #         print("skipping", filename)
    #         continue
    #     outpath = os.path.join("local-data/items/nuscenes-vis-set-1", filename.split("/")[-1])
    #     if os.path.exists(outpath):
    #         print("already saved", outpath)
    #         continue
    #     image_cv2 = cv2.imread(filename)
    #     vectorized = np.array([box.vectorize().cpu().numpy() for box in boxes])
    #     if len(vectorized.shape) > 2:
    #         vectorized = vectorized.squeeze(axis=1)
    #     classes = [0 for _ in boxes]
    #     vis_image = draw_boxes3d_cam(image_cv2, vectorized, classes, kitti_metadata, intrinsics)

    #     print(f"Saving to {outpath}")
    #     cv2.imwrite(outpath, vis_image)
