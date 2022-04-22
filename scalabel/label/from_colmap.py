import open3d as o3d
import numpy as np
import os


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
            points.append([x, y, z])

    return np.array(points, dtype=np.float64)


def transform_points(cam, points):
    # print("points shape", points.shape, points.dtype, points[:10])
    # print("cam translation", cam["translation"])
    return points - np.array(cam["translation"], dtype=np.float64)


def save_points(out_path, cam, points):
    img_name = cam["image_name"]
    file_name = f"{img_name.split('.')[0]}.ply"
    file_path = os.path.join(out_path, file_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)


if __name__ == "__main__":
    # in_file = "local-data/items/bdd100k/sfm/output/fused_0.ply"
    # from_colmap(in_file)

    # poses_path = "local-data/items/nuscenes-sfm/sparse/output/images.txt"
    # get_cam_poses(poses_path)

    """
    Todo:
    - read points
    - read camera positions
    - for each cam, transform points and save to output file.
    """
    # points_path = "local-data/items/nuscenes-sfm/sparse/output/points3D.txt"
    # poses_path = "local-data/items/nuscenes-sfm/sparse/output/images.txt"
    # out_path = "local-data/items/nuscenes-sfm/sparse/point-clouds"
    points_path = "local-data/items/bdd100k/sfm/output/sparse_0/output/points3D.txt"
    poses_path = "local-data/items/bdd100k/sfm/output/sparse_0/output/images.txt"
    out_path = "local-data/items/bdd100k/sfm/output/sparse_0/output/point-clouds"

    points = get_points(points_path)
    # min_dist = 100
    # for point in points:
    #     norm = np.linalg.norm(point)
    #     if norm < min_dist:
    #         min_dist = norm

    # print("min_dst", min_dist)
    cams = get_cam_poses(poses_path)

    for cam in cams:
        cam_points = transform_points(cam, points)
        print("points", cam_points[:3])
        save_points(out_path, cam, cam_points)
