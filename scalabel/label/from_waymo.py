"""Convert Waymo Open format dataset to Scalabel."""
import argparse
import os
import glob
import math
import numpy as np
from typing import List
from functools import partial
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2, utils

from ..common.parallel import pmap
from .io import save
from .typing import Frame, Label, ImageSize

classes_type2name = {
            label_pb2.Label.Type.TYPE_VEHICLE: "vehicle",
            label_pb2.Label.Type.TYPE_PEDESTRIAN: "pedestrian",
            label_pb2.Label.Type.TYPE_CYCLIST: "cyclist",
            label_pb2.Label.Type.TYPE_SIGN: "sign",
        }
cameras_id2name = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="waymo to scalabel")
    parser.add_argument(
        "--input",
        "-i",
        help="path to Waymo data (tfrecords).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".'",
        help="Output path for Scalabel format annotations.",
    )
    return parser.parse_args()


def cart2hom(pts_3d: np.ndarray) -> np.ndarray:
    """nx3 points in Cartesian to Homogeneous by appending 1"""
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def velo_to_rect(points, calib):
    axes_transform = np.array([[0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1]], dtype=np.float32)
    transform = np.matmul(axes_transform, np.linalg.inv(calib))
    return np.dot(cart2hom(points), transform.T)[:, :3]


def heading_velo_to_rect(heading, calib):
    # waymo heading given in lateral direction (negative)
    points = np.array([[0., 0., 0.], [-1., 0., 0.]])
    rot_mat = np.array([[np.cos(heading), -np.sin(heading), 0],
                        [np.sin(heading), np.cos(heading), 0],
                        [0, 0, 1]])
    points = np.dot(points, rot_mat.T)
    # first, transform to current camera coordinate system
    points = np.dot(cart2hom(points), np.linalg.inv(calib).T)
    # KITTI style heading can now be computed via:
    return math.atan2(points[1, 0] - points[0, 0], points[1, 1] - points[0, 1])


def parse_frame(frame, frame_id: int, output_dir: str) -> List[Frame]:
    lidar_labels, projected_lidar_labels, camera_labels, seq_token = frame.laser_labels, frame.projected_lidar_labels, \
                                                                      frame.camera_labels, frame.context.name
    frame_name = seq_token + "_{:07d}.jpg".format(frame_id)
    results = []
    for cam_idx, cam in enumerate(cams):
        proj_lidar_labels = [l for l in projected_lidar_labels if l.name == cam][0]

        # add image to coco
        intrinsics, extrinsics, img_size = calib[cam_idx]['intrinsic'], calib[cam_idx]['extrinsic'], calib[cam_idx]['img_wh']
        seq_dir = seq_token + '_' + camera_names[cam_idx].lower()
        img_filepath = os.path.join(output_dir, seq_dir, 'images', frame_name)

        # compute cam2global
        global_from_car = np.array(frame.pose.transform).reshape(4, 4)
        cam_to_global = np.dot(global_from_car, extrinsics)

        if not os.path.exists(img_filepath):
            img = imgs[cam_idx]
            save(img_filepath, img)  # TODO img save


        for i, cam_label in enumerate(proj_lidar_labels.labels):
            label_id = cam_label.id.replace(camera_names[cam_idx], '')[:-1]
            label = lidar_labels[[l.id for l in lidar_labels].index(label_id)]

            # Assign a category name to the label
            class_name = classes.get(label.type, None)
            if not class_name:
                continue

            # Transform the label position to the camera space.
            box3d = np.empty((7,), dtype=np.float32)
            ctr = np.array(
                [[label.box.center_x, label.box.center_y, label.box.center_z]])
            box3d[0:3] = velo_to_rect(ctr, extrinsics)[0]
            box3d[3:6] = label.box.height, label.box.width, label.box.length
            box3d[6] = heading_velo_to_rect(label.box.heading, extrinsics)
            box2d = np.array(
                [cam_label.box.center_x - cam_label.box.length / 2,
                 cam_label.box.center_y - cam_label.box.width / 2,
                 cam_label.box.length,
                 cam_label.box.width])

        frame = Frame(name=frame_name, video_name=seq_dir, frame_index=frame_id, size=img_size)
        results.append(frame)

    return results


def get_cams_from_frame(frame, cameras, decode_image=True):
    imgs = []
    calib = []
    for camera_name in cameras:
        camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)

        # Decode image and calibration
        if decode_image:
            camera = utils.get(frame.images, camera_name)
            imgs.append(camera.image)
        camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)
        camera_intrinsic = np.eye(3)
        camera_intrinsic[0, 0] = camera_calibration.intrinsic[0]
        camera_intrinsic[1, 1] = camera_calibration.intrinsic[1]
        camera_intrinsic[0, 2] = camera_calibration.intrinsic[2]
        camera_intrinsic[1, 2] = camera_calibration.intrinsic[3]
        img_size = ImageSize(height=camera_calibration.height, width=camera_calibration.width)
        calib.append({'extrinsic': camera_extrinsic, 'intrinsic': camera_intrinsic, 'img_size': img_size})

    return imgs, calib


def parse_record(output_dir: str, record_name: str) -> List[Frame]:
    """Parse data into List of Scalabel Label type per frame."""
    datafile = WaymoDataFileReader(record_name)
    table = datafile.get_record_table()
    frames = []
    for frame_id in range(len(table)):
        offset = table[frame_id]

        # jump to correct place before reading
        datafile.seek(offset)
        # read frame
        frame = datafile.read_record()

        # Get images, calibration, lidar and targets
        imgs, calib = get_cams_from_frame(frame, cams)

        # add images and annotations to coco
        frames.extend(parse_frame(frame, frame_id, output_dir))

    return frames


def from_waymo(data_path: str, output_dir: str) -> List[Frame]:
    """Function converting Waymo data to Scalabel format."""
    func = partial(parse_record, output_dir)
    frames = pmap(func, [filename for filename in glob.glob(data_path + "/*.tfrecord")])
    return frames


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    result = from_waymo(args.input, args.output)
    save(os.path.join(args.output, "scalabel_anns.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
