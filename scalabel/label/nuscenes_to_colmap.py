import numpy as np
import cv2
import os
import json
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

test_names = [
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151268662404.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151428162404.jpg",
]

# idxs = [77, 81]


def nuscenes_to_colmap(dataroot, version, img_out_file, cam_out_file, scene_idx=81):
    print("Version, scene", version, scene_idx)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    scene = nusc.scene[scene_idx]
    sample_token = scene["first_sample_token"]
    img_data = []
    while sample_token:
        sample = nusc.get("sample", sample_token)
        sensor = "CAM_FRONT"
        cam_front_data = nusc.get("sample_data", sample["data"][sensor])
        ego_pose_token = cam_front_data["ego_pose_token"]
        ego_pose = nusc.get("ego_pose", ego_pose_token)
        qw, qx, qy, qz = ego_pose["rotation"]
        tx, ty, tz = ego_pose["translation"]
        filename, _, _ = nusc.get_sample_data(cam_front_data["token"])
        sample_token = sample["next"]
        rotation = ego_pose["rotation"]
        translation = ego_pose["translation"]

        img_data.append((rotation, translation, filename.split("/")[-1]))

    sample_token = scene["first_sample_token"]
    sample = nusc.get("sample", sample_token)
    sensor = "CAM_FRONT"
    cam_front_data = nusc.get("sample_data", sample["data"][sensor])
    cs_record = nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
    intrinsics = cs_record["camera_intrinsic"]

    CAM_ID = 1
    # Write camera file
    width = cam_front_data["width"]
    height = cam_front_data["height"]
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    cam_line = f"{CAM_ID} PINHOLE {width} {height} {fx} {fy} {cx} {cy}"

    with open(cam_out_file, "w") as f:
        f.write(cam_line)

    # Write
    img_lines = []
    t_lines = []
    geo_lines = []
    for idx, (rotation, translation, filename) in enumerate(img_data):
        image_id = idx + 1
        # print("rotation", rotation)
        print("translation", translation)
        qw, qx, qy, qz = rotation
        tx, ty, tz = translation
        geo_lines.append(f"{filename} {tx} {ty} {tz}")
        t_lines.append(f"{tx},{ty}")
        q = Quaternion(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        q2 = Quaternion(axis=[0, 1, 0], angle=-np.pi / 2)
        q = q * q2
        R = q.rotation_matrix
        new_t = -R.T.dot(t)
        new_r = R.T
        new_q = Quaternion(matrix=new_r)
        qw, qx, qy, qz = new_q.elements
        tx, ty, tz = new_t
        print(new_r)
        img_line = f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {CAM_ID} {filename}"
        img_lines.append(img_line)
        img_lines.append("")

    with open("translation.txt", "w") as f:
        for line in t_lines:
            f.write(line)
            f.write("\n")

    with open(img_out_file, "w") as f:
        for line in img_lines:
            f.write(line)
            f.write("\n")

    with open("geo_reg.txt", "w") as f:
        for line in geo_lines:
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    dataroot = "local-data/items/nuscenes-set-1"
    version = "v1.0-trainval"
    # dataroot = "local-data/items/nuscenes-mini"
    # version = "v1.0-mini"
    img_out_file = "sfm-test/images.txt"
    cam_out_file = "sfm-test/cameras.txt"

    nuscenes_to_colmap(dataroot, version, img_out_file, cam_out_file, scene_idx=81)
