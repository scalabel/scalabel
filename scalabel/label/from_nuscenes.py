"""Conversion script for NuScenes to Scalabel."""
import argparse
import os
from datetime import datetime
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.parallel import NPROC, pmap
from ..common.typing import DictStrAny
from .io import save
from .typing import (
    Box2D,
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
    get_extrinsics_from_matrix,
    get_intrinsics_from_matrix,
    rotation_y_to_alpha,
)

try:
    import nuscenes as nu
    from nuscenes.eval.detection.constants import DETECTION_NAMES
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.nuscenes import NuScenes
    from nuscenes.scripts.export_2d_annotations_as_json import (
        post_process_coords,
    )
    from nuscenes.utils.data_classes import Box, Quaternion
    from nuscenes.utils.geometry_utils import (
        box_in_image,
        transform_matrix,
        view_points,
    )
    from nuscenes.utils.splits import create_splits_scenes
except ImportError:
    nu = None

cams = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="nuscenes to scalabel")
    parser.add_argument(
        "--input",
        "-i",
        help="path to NuScenes data root.",
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=["v1.0-trainval", "v1.0-test", "v1.0-mini"],
        help="NuScenes dataset version to convert.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for Scalabel format annotations.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def load_data(filepath: str, version: str) -> Tuple[nu.NuScenes, pd.DataFrame]:
    """Load nuscenes data and extract meta-information into dataframe."""
    data = NuScenes(version=version, dataroot=filepath, verbose=True)
    records = [
        (data.get("sample", record["first_sample_token"])["timestamp"], record)
        for record in data.scene
    ]
    entries = []

    for start_time, record in sorted(records):
        start_time = (
            data.get("sample", record["first_sample_token"])["timestamp"]
            / 1000000
        )
        token = record["token"]
        name = record["name"]
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record["name"].split("-")[:2])
        first_sample_token = record["first_sample_token"]

        entries.append((host, name, date, token, first_sample_token))

    dataframe = pd.DataFrame(
        entries,
        columns=[
            "host",
            "scene_name",
            "date",
            "scene_token",
            "first_sample_token",
        ],
    )
    return data, dataframe


def quaternion_to_yaw(quat: Quaternion, in_image_frame: bool = True) -> float:
    """Convert quaternion angle representation to yaw."""
    if in_image_frame:
        v = np.dot(quat.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
    else:
        v = np.dot(quat.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

    return yaw  # type: ignore


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle  to quaternion representation."""
    return Quaternion(
        scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
    ).elements


def transform_boxes(boxes: List[Box], pose: DictStrAny) -> None:
    """Move boxes from world space to given pose space.

    Note: mutates input boxes.
    """
    translation = -np.array(pose["translation"])
    rotation = Quaternion(pose["rotation"]).inverse

    for box in boxes:
        box.translate(translation)
        box.rotate(rotation)


def parse_labels(
    data: NuScenes,
    boxes: List[Box],
    ego_pose: DictStrAny,
    calibration_cam: Optional[DictStrAny] = None,
    img_size: Optional[Tuple[int, int]] = None,
) -> Optional[List[Label]]:
    """Parse NuScenes LiDAR labels into single camera."""
    if len(boxes):
        labels = []
        # transform into the camera coord system
        transform_boxes(boxes, ego_pose)
        if calibration_cam is not None:
            transform_boxes(boxes, calibration_cam)
            intrinsic_matrix = np.array(calibration_cam["camera_intrinsic"])

        for box in boxes:
            box_class = category_to_detection_name(box.name)
            in_image = True
            if calibration_cam is not None:
                assert img_size is not None
                in_image = box_in_image(box, intrinsic_matrix, img_size)

            if in_image and box_class is not None:
                xyz = tuple(box.center.tolist())
                w, l, h = box.wlh
                roty = quaternion_to_yaw(box.orientation)

                box2d = None
                if calibration_cam is not None:
                    # Project 3d box to 2d.
                    corners = box.corners()
                    corner_coords = (
                        view_points(corners, intrinsic_matrix, True)
                        .T[:, :2]
                        .tolist()
                    )
                    # Keep only corners that fall within the image.
                    x1, y1, x2, y2 = post_process_coords(corner_coords)
                    box2d = Box2D(x1=x1, y1=y1, x2=x2, y2=y2)

                instance_data = data.get("sample_annotation", box.token)
                # Attributes can be retrieved via instance_data and also the
                # category is more fine-grained than box_class.
                # This information could be stored in attributes if needed in
                # the future
                label = Label(
                    id=instance_data["instance_token"],
                    category=box_class,
                    box2d=box2d,
                    box3d=Box3D(
                        location=xyz,
                        dimension=(h, w, l),
                        orientation=(0, roty, 0),
                        alpha=rotation_y_to_alpha(roty, xyz),  # type: ignore
                    ),
                )
                labels.append(label)

        return labels
    return None


def ego_pose_to_extrinsics(ego_pose: DictStrAny) -> Extrinsics:
    """Convert NuScenes ego pose to extrinsics."""
    matrix = transform_matrix(
        ego_pose["translation"],
        Quaternion(ego_pose["rotation"]),
        inverse=False,
    )
    return get_extrinsics_from_matrix(matrix)


def calibration_to_intrinsics(calibration: DictStrAny) -> Intrinsics:
    """Convert calibration ego pose to Intrinsics."""
    matrix = np.array(calibration["camera_intrinsic"])
    return get_intrinsics_from_matrix(matrix)


def parse_sequence(
    data: NuScenes, scene_info: Tuple[str, str]
) -> Tuple[List[Frame], List[FrameGroup]]:
    """Parse a full NuScenes sequence and convert it into scalabel frames."""
    sample_token, scene_name = scene_info
    frames, groups = [], []
    frame_index = 0
    while sample_token:
        sample = data.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = data.get("sample_data", lidar_token)
        timestamp = lidar_data["timestamp"]
        lidar_filepath = lidar_data["filename"]
        ego_pose = data.get("ego_pose", lidar_data["ego_pose_token"])
        frame_names = []
        for cam in cams:
            cam_token = sample["data"][cam]
            cam_data = data.get("sample_data", cam_token)
            ego_pose_cam = data.get("ego_pose", cam_data["ego_pose_token"])
            cam_filepath = cam_data["filename"]
            img_wh = (cam_data["width"], cam_data["height"])
            calibration_cam = data.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )

            boxes = data.get_boxes(lidar_token)
            labels = parse_labels(
                data, boxes, ego_pose_cam, calibration_cam, img_wh
            )

            frame = Frame(
                name=os.path.basename(cam_filepath),
                videoName=scene_name,
                frameIndex=frame_index,
                url=cam_filepath,
                timestamp=cam_data["timestamp"],
                extrinsics=ego_pose_to_extrinsics(ego_pose_cam),
                intrinsics=calibration_to_intrinsics(calibration_cam),
                size=ImageSize(width=img_wh[0], height=img_wh[1]),
                labels=labels,
            )
            frame_names.append(frame.name)
            frames.append(frame)

        group = FrameGroup(
            name=sample_token,
            videoName=scene_name,
            frameIndex=frame_index,
            url=lidar_filepath,
            extrinsics=ego_pose_to_extrinsics(ego_pose),
            timestamp=timestamp,
            frames=frame_names,
            labels=parse_labels(data, data.get_boxes(lidar_token), ego_pose),
        )
        groups.append(group)

        sample_token = sample["next"]
        frame_index += 1

    return frames, groups


def from_nuscenes(
    data_path: str,
    version: str,
    split: str,
    output_dir: str,
    nproc: int = NPROC,
) -> Dataset:
    """Convert NuScenes dataset to Scalabel format."""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    data, df = load_data(data_path, version)
    scene_names_per_split = create_splits_scenes()

    first_sample_tokens = []
    for token, name in zip(df.first_sample_token.values, df.scene_name.values):
        if name in scene_names_per_split[split]:
            first_sample_tokens.append(token)

    func = partial(parse_sequence, data)
    if nproc > 1:
        partial_results = pmap(
            func,
            zip(first_sample_tokens, scene_names_per_split[split]),
            nprocs=nproc,
        )
    else:
        partial_results = map(  # type: ignore
            func,
            zip(first_sample_tokens, scene_names_per_split[split]),
        )
    frames, groups = [], []
    for f, g in partial_results:
        frames.extend(f)
        groups.extend(g)

    cfg = Config(categories=[Category(name=n) for n in DETECTION_NAMES])
    dataset = Dataset(frames=frames, groups=groups, config=cfg)
    return dataset


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    assert nu is not None, (
        "Please install the requirements in scripts/optional.txt to use"
        "NuScenes conversion."
    )

    if "mini" in args.version:
        splits_to_iterate = ["mini_train", "mini_val"]
    elif "test" in args.version:
        splits_to_iterate = ["test"]
    else:
        splits_to_iterate = ["train", "val"]

    for split in splits_to_iterate:
        result = from_nuscenes(
            args.input,
            args.version,
            split,
            args.output,
            args.nproc,
        )
        save(os.path.join(args.output, f"scalabel_{split}.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
