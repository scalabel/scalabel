import open3d as o3d
import numpy as np
import OpenEXR as exr
import Imath
import os
from pathlib import Path


def read_depth_exr_file(file_path):
    """
    From https://stackoverflow.com/a/64691893
    """
    exrfile = exr.InputFile(file_path.as_posix())
    raw_bytes = exrfile.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()["displayWindow"].max.y + 1 - exrfile.header()["displayWindow"].min.y
    width = exrfile.header()["displayWindow"].max.x + 1 - exrfile.header()["displayWindow"].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map


def get_depth_map(file_path):
    depth = read_depth_exr_file(Path(file_path))
    return depth


def get_point_cloud(depth, intrinsics, interval=4):
    fx, fy, cx, cy = intrinsics
    h, w = depth.shape
    points = []
    for i in range(w):
        for j in range(h):
            if i % interval != 0 or j % interval != 0:
                continue
            z = depth[j, i]
            x = (i - cx) * z / fx
            y = (j - cy) * z / fy
            # Swap axes from camera to lidar frame
            points.append([z, -x, -y])
    points = np.array(points)
    return points


def save_points(out_path, sample_idx, points):
    file_name = f"{sample_idx}.ply"
    file_path = os.path.join(out_path, file_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)


if __name__ == "__main__":
    for sample_idx in range(1080, 1090):
        print("sample_idx", sample_idx)
        file_path = f"/scratch-second/egroenewald/synscapes/Synscapes/img/depth/{sample_idx}.exr"
        outpath = "local-data/items/synscapes-sample/point-cloud"
        depth = get_depth_map(file_path)
        fx = 1590.83437
        fy = 1592.79032
        cx = 771.31406
        cy = 360.79945
        points = get_point_cloud(depth, (fx, fy, cx, cy))
        save_points(outpath, sample_idx, points)
