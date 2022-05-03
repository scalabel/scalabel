import open3d as o3d
import numpy as np
import os
import json
from pyquaternion import Quaternion
from operator import itemgetter


def get_cam_poses(file_path):
    cams = []
    with open(file_path) as file:
        count = 0
        for line in file:
            if line.startswith("#"):
                continue
            count += 1
            if count % 2 == 0:
                continue
            img_id, qw, qx, qy, qz, x, y, z, cam_id, name = line.split(" ")
            cam = {"translation": [x, y, z], "rotation": [qw, qx, qy, qz], "image_name": name.strip()}
            cams.append(cam)
    return cams


def get_points(file_path):
    points = []
    with open(file_path) as file:
        for line in file:
            if line.startswith("#"):
                continue
            point_id, x, y, z, r, g, b, *_ = line.split(" ")
            points.append([x, y, z, 1])

    return np.array(points, dtype=np.float64).T


def get_dense_points(file_path, filter=50):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    points = points[::filter]
    points = np.append(points, np.ones((points.shape[0], 1)), axis=1).T
    return points


def transform_points(cam, points):
    """
    Transform points using camera pose.
    Pose translation and rotation define the projection from world to
    camera coordinates.
    """
    q = Quaternion(np.array(cam["rotation"], dtype=np.float64))
    proj_mat = np.identity(4)
    proj_mat[:3, :3] = q.rotation_matrix
    position = np.array(cam["translation"], dtype=np.float64)
    proj_mat[:3, 3] = position
    projected = proj_mat.dot(points).T
    # Reorder axis
    projected = projected[:, [2, 0, 1, 3]]
    projected[:, 1:3] *= -1
    return projected[:, :3]


def filter_points(points, max_dist=60, max_height=5):
    norms = np.apply_along_axis(np.linalg.norm, 1, points)
    heights = points[:, 2]
    forward = points[:, 0]
    sideways = np.abs(points[:, 1])
    keep_idx = (norms < max_dist) & (heights < max_height) & (forward > 0) & (sideways < forward)
    print(f"Filtering from {len(points)} to {np.sum(keep_idx)}")
    return points[keep_idx]


def save_points(out_path, cam, points):
    img_name = cam["image_name"]
    file_name = f"{img_name.split('.')[0]}.ply"
    file_path = os.path.join(out_path, file_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)


def process(cams, point_func, points_path, outpath):
    points = point_func(points_path)
    # for idx in cam_indices:
    #     cam = cams[idx]
    for cam in cams:
        print(cam["image_name"])
        cam_points = filter_points(transform_points(cam, points))
        save_points(outpath, cam, cam_points)


def save_scalabel_file(cams, img_path, points_path, json_out, sensor_json_out, intrinsics):
    frames = []
    frame_groups = []

    for idx, cam in enumerate(cams):
        img_name = cam["image_name"]
        name = img_name.split(".")[0]
        points_name = f"{name}.ply"
        img_file = os.path.join(img_path, img_name)
        points_file = os.path.join(points_path, points_name)
        frames.append(
            {
                "name": f"img_{name}",
                "url": img_file.replace("local-data", "http://localhost:8686"),
            }
        )
        frames.append(
            {
                "name": f"pc_{name}",
                "url": points_file.replace("local-data", "http://localhost:8686"),
            }
        )
        frame_groups.append(
            {
                "name": name,
                "url": None,
                "videoName": None,
                "intrinsics": None,
                "extrinsics": None,
                "attributes": {},
                "timestamp": idx,
                "frameIndex": idx,
                "size": None,
                "frames": [f"img_{name}", f"pc_{name}"],
            }
        )

    sensor_data = [
        {
            "id": 0,
            "name": "image",
            "type": "image",
            "intrinsics": {
                "focal": intrinsics["focal"],
                "center": intrinsics["center"],
                "skew": 0,
                "radial": None,
                "tangential": None,
            },
            "extrinsics": None,
        },
        {
            "id": 1,
            "name": "pc",
            "type": "pointcloud",
            "extrinsics": {"location": [0.0, 0.0, 0.0], "rotation": [0.0, -1.5707963267948966, 1.5707963267948966]},
        },
    ]

    output_data = {"frames": frames, "frameGroups": frame_groups}
    with open(json_out, "w") as file:
        json.dump(output_data, file, indent=4)
    print(f"saved json to {json_out}")

    with open(sensor_json_out, "w") as file:
        json.dump(sensor_data, file, indent=4)
    print(f"saved sensor json to {sensor_json_out}")


if __name__ == "__main__":
    configs = {
        "kitti": {
            "sparse_points": "/Users/elrich/code/colmap/yutian/kitti_seq_1/sparse_gps/0/output/points3D.txt",
            "dense_points": "/Users/elrich/code/colmap/yutian/kitti_seq_1/fused.ply",
            "poses": "/Users/elrich/code/colmap/yutian/kitti_seq_1/sparse_gps/0/output/images.txt",
            "sparse_out": "local-data/items/sfm/kitti_seq_1/sparse",
            "dense_out": "local-data/items/sfm/kitti_seq_1/dense",
            "img_path": "local-data/items/kitti-tracking/kitti-sequence-1/0001",
            "json_out": "local-data/kitti_seq_1_sfm.json",
            "sensor_json_out": "local-data/kitti_seq_1_sfm_sensors.json",
            # "intrinsics": {
            #     "focal": [721.5377197265625, 721.5377197265625],
            #     "center": [609.559326171875, 172.85400390625],
            # },
            "intrinsics": {"focal": [705.5156980248752, 705.5156980248752], "center": [621.0, 187.5]},
        },
        "nuscenes": {
            "sparse_points": "/Users/elrich/code/colmap/yutian/nuscenes/sparse/0/output/points3D.txt",
            "dense_points": "/Users/elrich/code/colmap/yutian/nuscenes/fused.ply",
            "poses": "/Users/elrich/code/colmap/yutian/nuscenes/sparse/0/output/images.txt",
            "sparse_out": "local-data/items/sfm/nuscenes_81/sparse",
            "dense_out": "local-data/items/sfm/nuscenes_81/dense",
            "img_path": "local-data/items/nuscenes-set-1/nuscenes-sequence-81/full-sequence",
            "json_out": "local-data/nuscenes_81_sfm.json",
            "sensor_json_out": "local-data/nuscenes_81_sfm_sensors.json",
            "intrinsics": {
                "focal": [1262.1158922506606, 1262.1158922506606],
                "center": [800.0, 450.0],
            },
        },
    }
    config = configs["nuscenes"]

    cams = get_cam_poses(config["poses"])
    cams.sort(key=lambda c: c["image_name"])
    cam_indices = range(0, 20)
    cams = itemgetter(*cam_indices)(cams)
    print("len cams", len(cams))

    # process(cams, get_points, config["sparse_points"], config["sparse_out"])
    process(cams, get_dense_points, config["dense_points"], config["dense_out"])

    save_scalabel_file(
        cams,
        config["img_path"],
        config["dense_out"],
        # config["sparse_out"],
        config["json_out"],
        config["sensor_json_out"],
        config["intrinsics"],
    )
