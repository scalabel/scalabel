"""Convert Waymo Open format dataset to Scalabel."""
import argparse
import glob
import math
import os
from functools import partial
from typing import List, Tuple

import numpy as np

try:
    from simple_waymo_open_dataset_reader import (
        WaymoDataFileReader,
        dataset_pb2,
        label_pb2,
        utils,
    )

    # pylint: disable=no-member
    classes_type2name = {
        label_pb2.Label.Type.TYPE_VEHICLE: "vehicle",
        label_pb2.Label.Type.TYPE_PEDESTRIAN: "pedestrian",
        label_pb2.Label.Type.TYPE_CYCLIST: "cyclist",
        label_pb2.Label.Type.TYPE_SIGN: "sign",
    }
    # pylint: enable=no-member
except ImportError:
    WaymoDataFileReader = None

from ..common.parallel import pmap
from .io import save
from .typing import (
    Box2D,
    Box3D,
    Extrinsics,
    Frame,
    ImageSize,
    Intrinsics,
    Label,
)
from .utils import (
    get_extrinsics_from_matrix,
    get_matrix_from_extrinsics,
    get_matrix_from_intrinsics,
)

cameras_id2name = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}


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
        default="./",
        help="Output path for Scalabel format annotations.",
    )
    parser.add_argument(
        "--save_images",
        "-s",
        action="store_true",
        help="If the images should be extracted from .tfrecords and saved."
        "(necessary for using Waymo Open data with Scalabel format "
        "annotations)",
    )
    parser.add_argument(
        "--use_lidar_labels",
        action="store_true",
        help="If the conversion script should use the LiDAR labels as GT for "
        "conversion.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def cart2hom(pts_3d: np.ndarray) -> np.ndarray:
    """Nx3 points in Cartesian to Homogeneous by appending ones."""
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def project_points_to_image(
    points: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Project Nx3 points to Nx2 pixel coordinates with 3x3 intrinsics."""
    pts_3d_rect = cart2hom(points)
    campad = np.identity(4)
    campad[: intrinsics.shape[0], : intrinsics.shape[1]] = intrinsics
    pts_2d = np.dot(pts_3d_rect, np.transpose(campad))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, :2]  # type: ignore


def rotation_y_to_alpha(
    rotation_y: float, center_proj_x: float, focal_x: float, center_x: float
) -> float:
    """Convert rotation around y-axis to viewpoint angle (alpha)."""
    alpha = rotation_y - math.atan2(center_proj_x - center_x, focal_x)
    if alpha > math.pi:
        alpha -= 2 * math.pi
    if alpha <= -math.pi:
        alpha += 2 * math.pi
    return alpha


def points_transform(points: np.ndarray, calib: np.ndarray) -> np.ndarray:
    """Transform points from global to camera coordinate system."""
    axes_transform = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    transform = np.matmul(axes_transform, np.linalg.inv(calib))
    return np.dot(cart2hom(points), transform.T)[:, :3]  # type: ignore


def heading_transform(heading: float, calib: np.ndarray) -> float:
    """Transform heading from global to camera coordinate system."""
    # waymo heading given in lateral direction (negative)
    points = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    rot_mat = np.array(
        [
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1],
        ]
    )
    points = np.dot(points, rot_mat.T)
    # first, transform to current camera coordinate system
    points = np.dot(cart2hom(points), np.linalg.inv(calib).T)
    # KITTI style heading can now be computed via:
    return math.atan2(points[1, 0] - points[0, 0], points[1, 1] - points[0, 1])


def parse_lidar_labels(
    frame: dataset_pb2.Frame,
    intrinsics: Intrinsics,
    cam2car: Extrinsics,
    camera: str,
    camera_id: int,
) -> List[Label]:
    """Parse the LiDAR-based annotations."""
    labels = []
    proj_lidar_labels = [
        l.labels for l in frame.projected_lidar_labels if l.name == camera_id
    ][0]
    for label in proj_lidar_labels:
        laser_label_id = label.id.replace(camera, "")[:-1]
        laser_label = frame.laser_labels[
            [l.id for l in frame.laser_labels].index(laser_label_id)
        ]

        # Assign a category name to the label
        class_name = classes_type2name.get(label.type, None)
        if not class_name:
            continue

        # Transform the label position to the camera space.
        laser_box3d = laser_label.box
        center = np.array(
            [
                [
                    laser_box3d.center_x,
                    laser_box3d.center_y,
                    laser_box3d.center_z,
                ]
            ]
        )
        cam2car_mat = get_matrix_from_extrinsics(cam2car)
        intrinsics_mat = get_matrix_from_intrinsics(intrinsics)
        center = points_transform(center, cam2car_mat)
        center_proj = project_points_to_image(center, intrinsics_mat)[0]
        heading = heading_transform(laser_box3d.heading, cam2car_mat)
        dim = laser_box3d.height, laser_box3d.width, laser_box3d.length
        box3d = Box3D(
            orientation=(0.0, heading, 0.0),
            location=tuple(center[0].tolist()),
            dimension=dim,
            alpha=rotation_y_to_alpha(
                heading,
                center_proj[0],
                intrinsics.focal[0],
                intrinsics.center[0],
            ),
        )

        box2d = Box2D(
            x1=label.box.center_x - label.box.length / 2,
            y1=label.box.center_y - label.box.width / 2,
            x2=label.box.center_x + label.box.length / 2,
            y2=label.box.center_y + label.box.width / 2,
        )
        labels.append(
            Label(category=class_name, box2d=box2d, box3d=box3d, id=label.id)
        )
    return labels


def parse_camera_labels(
    frame: dataset_pb2.Frame, camera_id: int
) -> List[Label]:
    """Parse the camera-based annotations."""
    labels = []
    camera_labels = [
        l.labels for l in frame.camera_labels if l.name == camera_id
    ][0]
    for label in camera_labels:
        if not label:
            continue
        # Assign a category name to the label
        class_name = classes_type2name.get(label.type, None)
        if not class_name:
            continue

        box2d = Box2D(
            x1=label.box.center_x - label.box.length / 2,
            y1=label.box.center_y - label.box.width / 2,
            x2=label.box.center_x + label.box.length / 2,
            y2=label.box.center_y + label.box.width / 2,
        )
        labels.append(Label(category=class_name, box2d=box2d, id=label.id))

    return labels


def get_calibration(
    frame: dataset_pb2.Frame, camera_id: int
) -> Tuple[Extrinsics, Extrinsics, Intrinsics, ImageSize]:
    """Load and decode calibration data of camera in frame."""
    calib = utils.get(frame.context.camera_calibrations, camera_id)
    cam2car = np.array(calib.extrinsic.transform).reshape(4, 4)
    car2global = np.array(frame.pose.transform).reshape(4, 4)
    cam2global = np.dot(car2global, cam2car)
    extrinsics = get_extrinsics_from_matrix(cam2global)
    extrinsics_local = get_extrinsics_from_matrix(cam2car)
    intrinsics = Intrinsics(
        focal=(calib.intrinsic[0], calib.intrinsic[1]),
        center=(calib.intrinsic[2], calib.intrinsic[3]),
    )
    image_size = ImageSize(height=calib.height, width=calib.width)
    return extrinsics, extrinsics_local, intrinsics, image_size


def parse_frame(
    frame: dataset_pb2.Frame,
    frame_id: int,
    output_dir: str,
    save_images: bool = False,
    use_lidar_labels: bool = False,
) -> List[Frame]:
    """Parse information in single frame to Scalabel Frame per camera."""
    frame_name = frame.context.name + "_{:07d}.jpg".format(frame_id)
    results = []
    for camera_id, camera in cameras_id2name.items():
        cam2global, cam2car, intrinsics, image_size = get_calibration(
            frame, camera_id
        )
        seq_dir = frame.context.name + "_" + camera.lower()
        img_filepath = os.path.join(output_dir, seq_dir, frame_name)

        if save_images and not os.path.exists(img_filepath):
            if not os.path.exists(os.path.dirname(img_filepath)):
                os.mkdir(os.path.dirname(img_filepath))
            im_bytes = utils.get(frame.images, camera_id).image
            open(img_filepath, "wb").write(im_bytes)

        if use_lidar_labels:
            labels = parse_lidar_labels(
                frame, intrinsics, cam2car, camera, camera_id
            )
        else:
            labels = parse_camera_labels(frame, camera_id)
        f = Frame(
            name=frame_name,
            video_name=seq_dir,
            frame_index=frame_id,
            size=image_size,
            extrinsics=cam2global,
            intrinsics=intrinsics,
            labels=labels,
        )
        results.append(f)

    return results


def parse_record(
    output_dir: str,
    save_images: bool,
    use_lidar_labels: bool,
    record_name: str,
) -> List[Frame]:
    """Parse data into List of Scalabel format annotations."""
    datafile = WaymoDataFileReader(record_name)
    table = datafile.get_record_table()
    frames = []
    for frame_id, offset in enumerate(table):
        # jump to correct place, read frame
        datafile.seek(offset)
        frame = datafile.read_record()

        # add images and annotations to coco
        frame = parse_frame(
            frame, frame_id, output_dir, save_images, use_lidar_labels
        )
        frames.extend(frame)

    return frames


def from_waymo(
    data_path: str,
    output_dir: str,
    save_images: bool = False,
    use_lidar_labels: bool = False,
    nproc: int = 4,
) -> List[Frame]:
    """Function converting Waymo data to Scalabel format."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    func = partial(parse_record, output_dir, save_images, use_lidar_labels)
    partial_frames = pmap(
        func,
        (filename for filename in glob.glob(data_path + "/*.tfrecord")),
        nprocs=nproc,
    )
    frames = []
    for f in partial_frames:
        frames.extend(f)
    return frames


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    assert WaymoDataFileReader is not None, (
        "Please install the requirements in scripts/optional.txt to use"
        "Waymo conversion."
    )
    result = from_waymo(
        args.input,
        args.output,
        args.save_images,
        args.use_lidar_labels,
        args.nproc,
    )
    save(os.path.join(args.output, "scalabel_anns.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
