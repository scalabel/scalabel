import open3d as o3d
import numpy as np
import os
from pyquaternion import Quaternion


def from_colmap(in_file):
    pcd = o3d.io.read_point_cloud(in_file)
    point_cloud_np = np.asarray(pcd.points)
    print(point_cloud_np.shape)
    for point in point_cloud_np[:100]:
        print(point)


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
            cam = {"translation": [x, y, z], "rotation": [qw, qx, qy, qz], "image_name": name}
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


def get_dense_points(file_path, filter=10):
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


def filter_points(points, max_dist=20):
    norms = np.apply_along_axis(np.linalg.norm, 1, points)
    keep_idx = np.where(norms < max_dist)
    return points[keep_idx]


def save_points(out_path, cam, points):
    img_name = cam["image_name"]
    file_name = f"{img_name.split('.')[0]}.ply"
    file_path = os.path.join(out_path, file_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)


if __name__ == "__main__":
    # points_path = "local-data/items/nuscenes-sfm/sparse/output/points3D.txt"
    # poses_path = "local-data/items/nuscenes-sfm/sparse/output/images.txt"
    # out_path = "local-data/items/nuscenes-sfm/sparse/point-clouds"
    points_path = "local-data/items/bdd100k/sfm/output/sparse_0/output/points3D.txt"
    poses_path = "local-data/items/bdd100k/sfm/output/sparse_0/output/images.txt"
    out_path = "local-data/items/bdd100k/sfm/output/sparse_0/output/point-clouds"
    dense_out_path = "local-data/items/bdd100k/sfm/output/dense-point-clouds"
    dense_points_path = "local-data/items/bdd100k/sfm/output/fused_0.ply"

    dense_points = get_dense_points(dense_points_path)
    # dense_points = dense_points[::10]
    # print(dense_points.shape)
    # dense_points = filter_points(dense_points)
    # print(dense_points.shape)

    cams = get_cam_poses(poses_path)

    cams.sort(key=lambda c: c["image_name"])
    imgs = [
        "2ef923e9-5d8874dd-0000481.jpg",
        "2ef923e9-5d8874dd-0000487.jpg",
        "2ef923e9-5d8874dd-0000493.jpg",
        "2ef923e9-5d8874dd-0000499.jpg",
        "2ef923e9-5d8874dd-0000505.jpg",
    ]
    for cam in cams:
        if cam["image_name"].strip() not in imgs:
            continue
        print(cam["image_name"])
        print(dense_points.shape)
        cam_points = transform_points(cam, dense_points)
        print(cam_points.shape)
        cam_points = filter_points(cam_points)
        print(cam_points.shape)
        save_points(dense_out_path, cam, cam_points)

    # points = get_points(points_path)
    # print("len cams", len(cams))

    # cams.sort(key=lambda c: c["image_name"])
    # for cam in cams:
    #     print(cam["image_name"])
    #     cam_points = transform_points(cam, points)
    #     save_points(out_path, cam, cam_points)
