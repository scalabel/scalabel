import numpy as np
import cv2
import os
import json
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from pyproj import Proj

test_names = [
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151268662404.jpg",
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151428162404.jpg",
]

# idxs = [77, 81]


def nuscenes_to_colmap(dataroot, version, img_out_file, cam_out_file, img_path, scene_idx=81):
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
        filename, _, _ = nusc.get_sample_data(cam_front_data["token"])
        sample_token = sample["next"]
        rotation = ego_pose["rotation"]
        tx, ty, tz = ego_pose["translation"]
        translation = [tx, -ty, tz]

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
    intrinsic_json = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    cam_line = f"{CAM_ID} PINHOLE {width} {height} {fx} {fy} {cx} {cy}"

    with open(cam_out_file, "w") as f:
        f.write(cam_line)

    with open(f"intrinsics-{scene_idx}.json", "w") as f:
        json.dump(intrinsic_json, f, indent=4)

    print("width, height", width, height)
    print("intrinsics", intrinsics)
    for i in range(10):
        sample_token = sample["next"]
        sample = nusc.get("sample", sample_token)
        sensor = "CAM_FRONT"
        cam_front_data = nusc.get("sample_data", sample["data"][sensor])
        cs_record = nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
        intrinsics = cs_record["camera_intrinsic"]
    print("intrinsics[10]", intrinsics)
    # Write
    """
    Todo:
    - Get a list of translations
    - Get a list of sample filenames and list of all filenames.
    - Read through both lists together, interpolating as you go
    - At the end, extrapolate the endings.
    """
    print("length", len(img_data))
    all_filenames = os.listdir(img_path)
    all_filenames.sort()
    translations = [t for _, t, _ in img_data]
    filenames = [f for _, _, f in img_data]
    all_translations = [0.0 for f in all_filenames]
    indices = []
    for f in filenames:
        indices.append(all_filenames.index(f))
    for idx, t in zip(indices, translations):
        all_translations[idx] = t
    # Handle interpolation
    for i in range(len(indices) - 1):
        t1 = np.array(translations[i])
        t2 = np.array(translations[i + 1])
        diff = t2 - t1
        num_indices = indices[i + 1] - indices[i]
        delta = diff / num_indices
        for j in range(1, num_indices):
            curr_idx = indices[i] + j
            all_translations[curr_idx] = t1 + delta * j
    # Handle extrapolation
    t1 = np.array(translations[0])
    t2 = np.array(translations[1])
    n = len(translations)
    tn1 = np.array(translations[n - 1])
    tn2 = np.array(translations[n - 2])
    start_delta = (t2 - t1) / (indices[1] - indices[0])
    for i in range(indices[0]):
        all_translations[i] = t1 - start_delta * (indices[0] - i)
    end_delta = (tn1 - tn2) / (indices[n - 1] - indices[n - 2])
    for i in range(indices[n - 1] + 1, len(all_translations)):
        all_translations[i] = tn1 + i * end_delta

    locations = []
    csv_lines = []
    csv_lines.append("name,latitude,longitude")
    proj = Proj(proj="merc", ellps="WGS84", preserve_units=False)
    nx, ny = proj(0, 0)
    null_island = np.array([nx, ny, 0], dtype=np.double)
    t0 = np.array(all_translations[0], dtype=np.double)
    for idx, translation in enumerate(all_translations):
        t = np.array(translation, dtype=np.double)
        t_rel = null_island + t - t0
        t_rel_x, t_rel_y, _ = t_rel
        lat, lon = proj(t_rel_x, t_rel_y, inverse=True)
        location = {
            "latitude": lat,
            "longitude": lon,
            "timestamp": idx,
            "speed": 0.0,
            "accuracy": 10.0,
            "course": 0.0,
        }
        locations.append(location)
        csv_lines.append(f"{idx},{lat},{lon}")

    location_data = {"locations": locations}
    # json.encoder.FLOAT_REPR = lambda x: format(x, '.5f')
    with open(f"locations-{scene_idx}.json", "w") as f:
        json.dump(location_data, f, indent=4)

    with open(f"locations-{scene_idx}_csv.csv", "w") as f:
        for line in csv_lines:
            f.write(line)
            f.write("\n")
    print("saved locations")


# def kitti_metadata(filepath):
#     locations = []
#     ts = 1
#     with open(filepath) as f:
#         for line in f:
#             vals = line.split(" ")
#             location = {
#                 "latitude": vals[0],
#                 "longitude": vals[1],
#                 "timestamp": ts,
#                 "speed": 0.0,
#                 "accuracy": 10.0,
#                 "course": 0.0,
#             }
#             locations.append(location)
#             ts += 1

#     return {"locations": locations}


# if __name__ == "__main__":
#     filepath = "local-data/items/kitti-tracking/oxts/training/oxts/0001.txt"
#     data = kitti_metadata(filepath)
#     outpath = "locations-0001.json"
#     with open(outpath, "w") as f:
#         json.dump(data, f)


if __name__ == "__main__":
    dataroot = "local-data/items/nuscenes-set-1"
    version = "v1.0-trainval"
    img_out_file = "sfm-test/images.txt"
    cam_out_file = "sfm-test/cameras.txt"
    # location_outpath = "local-data/items/nuscenes-set-1/nuscenes-85/locations.json"
    scene_idx = 85
    img_path = f"local-data/items/nuscenes-set-1/nuscenes-{scene_idx}/images"

    nuscenes_to_colmap(dataroot, version, img_out_file, cam_out_file, img_path, scene_idx=scene_idx)
