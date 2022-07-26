"""Conversion script from Scalabel to Waymo format."""
import os
import argparse
import json
import operator
from typing import Dict, List
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..common.io import open_write_text
from ..common.parallel import NPROC
from ..common.typing import DictStrAny, NDArrayF64
from .io import load
from .typing import Dataset
from .utils import (
    get_box_transformation_matrix,
    get_extrinsics_from_matrix,
    get_matrix_from_extrinsics,
)

try:
    import tensorflow.compat.v1 as tf

    tf.enable_eager_execution()
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

    obj_type_dict = {
        "pedestrian": label_pb2.Label.TYPE_PEDESTRIAN,
        "cyclist": label_pb2.Label.TYPE_CYCLIST,
        "vehicle": label_pb2.Label.TYPE_VEHICLE,
    }
    Waymo_INSTALL = True
except ImportError:
    Waymo_INSTALL = False
import pdb

lasers_name2id = {
    "FRONT": 0,
    "REAR": 1,
    "SIDE_LEFT": 2,
    "SIDE_RIGHT": 3,
    "TOP": 4,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Scalabel to Waymo 3D format")
    parser.add_argument(
        "--input",
        "-i",
        help=(
            "root directory of Scalabel label Json files or path to a label "
            "json file"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="path to save Waymo formatted label file",
    )
    parser.add_argument(
        "--data_path",
        help="Waymo dataset path",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def to_waymo(dataset: Dataset, data_path: str):
    """Convert to Waymo format."""
    objects = metrics_pb2.Objects()

    assert dataset.frames is not None

    waymo_dataset = tf.data.TFRecordDataset(
        [
            os.path.join(data_path, p)
            for p in os.listdir(data_path)
            if p.endswith(".tfrecord")
        ],
        compression_type="",
    )

    for i, data in enumerate(tqdm(waymo_dataset)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        scalabel_frame = dataset.frames[i]

        lidar2car_mat: NDArrayF64 = np.array(
            frame.context.laser_calibrations[
                lasers_name2id["TOP"]
            ].extrinsic.transform,
            dtype=np.float64,
        ).reshape(4, 4)

        for label in scalabel_frame.labels:
            o = metrics_pb2.Object()
            o.context_name = frame.context.name
            o.frame_timestamp_micros = frame.timestamp_micros

            translation = (
                np.dot(lidar2car_mat[:3, :3], label.box3d.location)
                + lidar2car_mat[:3, 3]
            )

            # Using extrinsic rotation here to align with Pytorch3D
            rot = R.from_euler("XYZ", label.box3d.orientation).as_matrix()
            rot = np.dot(lidar2car_mat[:3, :3], rot)
            rot = R.from_matrix(rot).as_euler("XYZ")
            _, _, yaw = rot.tolist()

            box = label_pb2.Label.Box()
            box.center_x = translation[0]
            box.center_y = translation[1]
            box.center_z = translation[2]
            box.length = label.box3d.dimension[2]
            box.width = label.box3d.dimension[1]
            box.height = label.box3d.dimension[0]
            box.heading = yaw
            o.object.box.CopyFrom(box)

            o.score = label.score
            o.object.id = label.id
            o.object.type = obj_type_dict[label.category]

            o.object.num_lidar_points_in_box = 100

            objects.objects.append(o)
    return objects


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    assert (
        Waymo_INSTALL is not None
    ), "Please install the waymo-open-dataset to use Waymo conversion."

    dataset = load(args.input, args.nproc)

    waymo_objects = to_waymo(dataset, args.data_path)

    # Write objects to a file.
    with open(args.output, "wb") as f:
        f.write(waymo_objects.SerializeToString())


if __name__ == "__main__":
    run(parse_arguments())
