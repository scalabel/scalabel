"""Convert Waymo Open format dataset to Scalabel."""
import argparse
import glob
import math
import os
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..common.parallel import NPROC
from ..common.typing import NDArrayF64
from ..label.transforms import xyxy_to_box2d
from ..label.utils import cart2hom, rotation_y_to_alpha

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
    Box3D,
    Category,
    Config,
    Dataset,
    Extrinsics,
    Frame,
    FrameGroup,
    ImageSize,
    Intrinsics,
    Label,
)
from .utils import (
    get_box_transformation_matrix,
    get_extrinsics_from_matrix,
    get_matrix_from_extrinsics,
)

cameras_id2name = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

lasers_name2id = {
    "FRONT": 0,
    "REAR": 1,
    "SIDE_LEFT": 2,
    "SIDE_RIGHT": 3,
    "TOP": 4,
}

waymo2kitti_RT: NDArrayF64 = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


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
        help="Output path for Scalabel format annotations.",
    )
    parser.add_argument(
        "--save-images",
        "-s",
        action="store_true",
        help="If the images should be extracted from .tfrecords and saved."
        "(necessary for using Waymo Open data with Scalabel format "
        "annotations)",
    )
    parser.add_argument(
        "--use-lidar-labels",
        action="store_true",
        help="If the conversion script should use the LiDAR labels as GT for "
        "conversion.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def heading_transform(laser_box3d: label_pb2, calib: NDArrayF64) -> float:
    """Transform heading from global to camera coordinate system."""
    rot_y = laser_box3d.heading
    transform_box_to_cam: NDArrayF64 = calib @ get_box_transformation_matrix(
        (
            laser_box3d.center_x,
            laser_box3d.center_y,
            laser_box3d.center_z,
        ),
        (laser_box3d.height, laser_box3d.length, laser_box3d.width),
        rot_y,
    )
    pt1: NDArrayF64 = np.array([-0.5, 0.5, 0, 1.0], dtype=np.float64)
    pt2: NDArrayF64 = np.array([0.5, 0.5, 0, 1.0], dtype=np.float64)
    pt1 = np.matmul(transform_box_to_cam, pt1).tolist()
    pt2 = np.matmul(transform_box_to_cam, pt2).tolist()
    return -math.atan2(pt2[2] - pt1[2], pt2[0] - pt1[0])


def parse_lidar_labels(
    frame: dataset_pb2.Frame,
    calib: Extrinsics,
    camera: Optional[str] = None,
    camera_id: Optional[int] = None,
) -> List[Label]:
    """Parse the LiDAR-based annotations."""
    labels = []
    if camera is None and camera_id is None:
        lidar2car = get_matrix_from_extrinsics(calib)
        car2lidar = np.linalg.inv(lidar2car)
        for label in frame.laser_labels:
            # Assign a category name to the label
            class_name = classes_type2name.get(label.type, None)
            if not class_name:
                continue

            laser_box3d = label.box
            center: NDArrayF64 = np.array(
                [
                    [
                        laser_box3d.center_x,
                        laser_box3d.center_y,
                        laser_box3d.center_z,
                    ]
                ],
                dtype=np.float64,
            )
            center_lidar = tuple(
                np.dot(car2lidar, cart2hom(center).T)[:3, 0].tolist()
            )
            heading = heading_transform(laser_box3d, car2lidar)
            dim = laser_box3d.height, laser_box3d.width, laser_box3d.length
            box3d = Box3D(
                orientation=(0.0, heading, 0.0),
                location=center_lidar,
                dimension=dim,
                alpha=rotation_y_to_alpha(heading, center_lidar),  # type: ignore # pylint: disable=line-too-long
            )
            labels.append(
                Label(
                    category=class_name, box2d=None, box3d=box3d, id=label.id
                )
            )
    else:
        cam2car = get_matrix_from_extrinsics(calib)
        car2cam = np.linalg.inv(cam2car)
        proj_lidar_labels = [
            l.labels
            for l in frame.projected_lidar_labels
            if l.name == camera_id
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
            center_cam = tuple(
                np.dot(car2cam, cart2hom(center).T)[:3, 0].tolist()
            )
            heading = heading_transform(laser_box3d, car2cam)
            dim = laser_box3d.height, laser_box3d.width, laser_box3d.length
            box3d = Box3D(
                orientation=(0.0, heading, 0.0),
                location=center_cam,
                dimension=dim,
                alpha=rotation_y_to_alpha(heading, center_cam),  # type: ignore
            )

            box2d = xyxy_to_box2d(
                label.box.center_x - label.box.length / 2,
                label.box.center_y - label.box.width / 2,
                label.box.center_x + label.box.length / 2,
                label.box.center_y + label.box.width / 2,
            )
            labels.append(
                Label(
                    category=class_name, box2d=box2d, box3d=box3d, id=label.id
                )
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

        box2d = xyxy_to_box2d(
            label.box.center_x - label.box.length / 2,
            label.box.center_y - label.box.width / 2,
            label.box.center_x + label.box.length / 2,
            label.box.center_y + label.box.width / 2,
        )
        labels.append(Label(category=class_name, box2d=box2d, id=label.id))

    return labels


def parse_frame_attributes(
    frame: dataset_pb2.Frame,
    use_lidar_labels: bool = False,
) -> Dict[str, Union[str, float]]:
    """Parse the camera-based attributes."""
    check_attribute = lambda x: x if x else "undefined"
    s = frame.context.stats

    attributes = {
        "time_of_day": check_attribute(s.time_of_day),
        "weather": check_attribute(s.weather),
        "location": check_attribute(s.location),
    }

    ocs = s.laser_object_counts if use_lidar_labels else s.camera_object_counts
    sensor = "laser" if use_lidar_labels else "camera"
    for oc in ocs:
        o_name = classes_type2name[oc.type]
        attribute_name = f"{sensor}_{o_name}_counts"
        attributes[attribute_name] = oc.count

    return attributes


def get_calibration(
    frame: dataset_pb2.Frame, camera_id: int
) -> Tuple[ImageSize, Intrinsics, Extrinsics, Extrinsics, Extrinsics]:
    """Load and decode calibration data of camera in frame."""
    calib = utils.get(frame.context.camera_calibrations, camera_id)

    image_size = ImageSize(height=calib.height, width=calib.width)

    intrinsics = Intrinsics(
        focal=(calib.intrinsic[0], calib.intrinsic[1]),
        center=(calib.intrinsic[2], calib.intrinsic[3]),
    )

    cam2car_mat: NDArrayF64 = np.array(
        calib.extrinsic.transform, dtype=np.float64
    ).reshape(4, 4)
    car2cam_mat = np.linalg.inv(cam2car_mat)
    car2cam_mat = np.dot(waymo2kitti_RT, car2cam_mat)
    cam2car_mat = np.linalg.inv(car2cam_mat)

    car2global_mat: NDArrayF64 = np.array(
        frame.pose.transform, dtype=np.float64
    ).reshape(4, 4)
    cam2global_mat = np.dot(car2global_mat, cam2car_mat)

    cam2car = get_extrinsics_from_matrix(cam2car_mat)
    car2global = get_extrinsics_from_matrix(car2global_mat)
    cam2global = get_extrinsics_from_matrix(cam2global_mat)

    return (
        image_size,
        intrinsics,
        cam2car,
        car2global,
        cam2global,
    )


def parse_frame(
    frame: dataset_pb2.Frame,
    frame_id: int,
    output_dir: str,
    save_images: bool = False,
    use_lidar_labels: bool = False,
) -> Tuple[List[Frame], List[FrameGroup]]:
    """Parse information in single frame to Scalabel Frame per camera."""
    frame_name = frame.context.name + f"_{frame_id:07d}.jpg"
    attributes = parse_frame_attributes(frame, use_lidar_labels)

    results, group_results = [], []
    frame_names = []
    sequence = frame.context.name
    for camera_id, camera in cameras_id2name.items():
        (
            image_size,
            intrinsics,
            cam2car,
            car2global,
            cam2global,
        ) = get_calibration(frame, camera_id)
        url = os.path.join(sequence, camera, frame_name)
        img_filepath = os.path.join(output_dir, sequence, camera, frame_name)

        img_name = (
            frame.context.name + "_" + camera.lower() + f"_{frame_id:07d}.jpg"
        )

        if save_images and not os.path.exists(img_filepath):
            if not os.path.exists(os.path.dirname(img_filepath)):
                os.makedirs(os.path.dirname(img_filepath))
            im_bytes = utils.get(frame.images, camera_id).image
            with open(img_filepath, "wb") as fp:
                fp.write(im_bytes)

        if use_lidar_labels:
            labels = parse_lidar_labels(frame, cam2car, camera, camera_id)
        else:
            labels = parse_camera_labels(frame, camera_id)

        f = Frame(
            name=img_name,
            videoName=sequence,
            frameIndex=frame_id,
            url=url,
            size=image_size,
            extrinsics=cam2global,
            intrinsics=intrinsics,
            labels=labels,
            attributes=attributes,
        )
        frame_names.append(img_name)
        results.append(f)

    url = f"segment-{frame.context.name}_with_camera_labels.tfrecord"

    lidar2car_mat: NDArrayF64 = np.array(
        frame.context.laser_calibrations[
            lasers_name2id["TOP"]
        ].extrinsic.transform,
        dtype=np.float64,
    ).reshape(4, 4)
    lidar2global_mat = np.dot(
        get_matrix_from_extrinsics(car2global), lidar2car_mat
    )

    lidar2car = get_extrinsics_from_matrix(lidar2car_mat)
    lidar2global = get_extrinsics_from_matrix(lidar2global_mat)

    group_results = [
        FrameGroup(
            name=frame_name,
            videoName=sequence,
            frameIndex=frame_id,
            url=url,
            extrinsics=lidar2global,
            frames=frame_names,
            labels=parse_lidar_labels(frame, lidar2car),
        )
    ]

    return results, group_results


def parse_record(
    output_dir: str,
    save_images: bool,
    use_lidar_labels: bool,
    record_name: str,
) -> Tuple[List[Frame], List[FrameGroup]]:
    """Parse data into List of Scalabel format annotations."""
    datafile = WaymoDataFileReader(record_name)
    table = datafile.get_record_table()
    frames, groups = [], []
    for frame_id, offset in enumerate(table):
        # jump to correct place, read frame
        datafile.seek(offset)
        frame = datafile.read_record()

        # add images and annotations to coco
        frame, group = parse_frame(
            frame, frame_id, output_dir, save_images, use_lidar_labels
        )
        frames.extend(frame)
        groups.extend(group)

    return frames, groups


def from_waymo(
    data_path: str,
    output_dir: str,
    save_images: bool = False,
    use_lidar_labels: bool = False,
    nproc: int = NPROC,
) -> Dataset:
    """Function converting Waymo data to Scalabel format."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    func = partial(parse_record, output_dir, save_images, use_lidar_labels)
    if nproc > 1:
        partial_results = pmap(
            func,
            (filename for filename in glob.glob(data_path + "/*.tfrecord")),
            nprocs=nproc,
        )
    else:
        partial_results = map(  # type: ignore
            func,
            (filename for filename in glob.glob(data_path + "/*.tfrecord")),
        )
    frames, groups = [], []
    for f, g in partial_results:
        frames.extend(f)
        groups.extend(g)

    cfg = Config(
        categories=[Category(name=n) for n in classes_type2name.values()]
    )
    dataset = Dataset(frames=frames, groups=groups, config=cfg)
    return dataset


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    assert WaymoDataFileReader is not None, (
        "Please install the requirements in scripts/optional.txt to use"
        "Waymo conversion."
    )
    if not os.path.exists(args.output):
        os.makedirs(args.output)
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
