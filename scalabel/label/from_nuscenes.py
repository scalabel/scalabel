"""Conversion script for NuScenes to Scalabel."""
import argparse
import os
from datetime import datetime
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.parallel import NPROC, pmap
from ..common.typing import DictStrAny, NDArrayF64
from ..label.transforms import xyxy_to_box2d
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
    get_extrinsics_from_matrix,
    get_intrinsics_from_matrix,
    rotation_y_to_alpha,
)

try:
    from nuscenes import NuScenes
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
    NuScenes = None

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
        default=None,
        help="Output path for Scalabel format annotations.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[],
        help="Data splits to convert.",
    )
    parser.add_argument(
        "--add-non-key",
        action="store_true",
        help="Add non-key frames (not annotated) to the converted data.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def load_data(filepath: str, version: str) -> Tuple[NuScenes, pd.DataFrame]:
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


def transform_boxes(
    boxes: List[Box], ego_pose: DictStrAny, car_from_sensor: DictStrAny
) -> None:
    """Move boxes from world space to given sensor frame.

    Note: mutates input boxes.
    """
    translation_car: NDArrayF64 = -np.array(
        ego_pose["translation"], dtype=np.float64
    )
    rotation_car = Quaternion(ego_pose["rotation"]).inverse
    translation_sensor: NDArrayF64 = -np.array(
        car_from_sensor["translation"], dtype=np.float64
    )
    rotation_sensor = Quaternion(car_from_sensor["rotation"]).inverse
    for box in boxes:
        box.translate(translation_car)
        box.rotate(rotation_car)
        box.translate(translation_sensor)
        box.rotate(rotation_sensor)


def parse_labels(
    data: NuScenes,
    boxes: List[Box],
    ego_pose: DictStrAny,
    calib_sensor: DictStrAny,
    img_size: Optional[Tuple[int, int]] = None,
    in_image_frame: bool = True,
) -> Optional[List[Label]]:
    """Parse NuScenes labels into sensor frame."""
    if len(boxes):
        labels = []
        # transform into the sensor coord system
        transform_boxes(boxes, ego_pose, calib_sensor)
        intrinsic_matrix: NDArrayF64 = np.array(
            calib_sensor["camera_intrinsic"], dtype=np.float64
        )
        for box in boxes:
            box_class = category_to_detection_name(box.name)
            in_image = True
            if img_size is not None:
                in_image = box_in_image(box, intrinsic_matrix, img_size)

            if in_image and box_class is not None:
                xyz = tuple(box.center.tolist())
                w, l, h = box.wlh
                roty = quaternion_to_yaw(box.orientation, in_image_frame)

                box2d = None
                if img_size is not None:
                    # Project 3d box to 2d.
                    corners = box.corners()
                    corner_coords = (
                        view_points(corners, intrinsic_matrix, True)
                        .T[:, :2]
                        .tolist()
                    )
                    # Keep only corners that fall within the image, transform
                    box2d = xyxy_to_box2d(*post_process_coords(corner_coords))

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


def get_extrinsics(
    ego_pose: DictStrAny, car_from_sensor: DictStrAny
) -> Extrinsics:
    """Convert NuScenes ego pose / sensor_to_car to global extrinsics."""
    global_from_car = transform_matrix(
        ego_pose["translation"],
        Quaternion(ego_pose["rotation"]),
        inverse=False,
    )
    car_from_sensor_ = transform_matrix(
        car_from_sensor["translation"],
        Quaternion(car_from_sensor["rotation"]),
        inverse=False,
    )
    extrinsics = np.dot(global_from_car, car_from_sensor_)
    return get_extrinsics_from_matrix(extrinsics)


def calibration_to_intrinsics(calibration: DictStrAny) -> Intrinsics:
    """Convert calibration ego pose to Intrinsics."""
    matrix: NDArrayF64 = np.array(
        calibration["camera_intrinsic"], dtype=np.float64
    )
    return get_intrinsics_from_matrix(matrix)


def parse_frame(
    data: NuScenes,
    scene_name: str,
    frame_index: int,
    cam_token: str,
    boxes: Optional[List[Box]] = None,
) -> Tuple[Frame, Optional[str]]:
    """Parse a single camera frame."""
    cam_data = data.get("sample_data", cam_token)
    ego_pose_cam = data.get("ego_pose", cam_data["ego_pose_token"])
    cam_filepath = cam_data["filename"]
    img_wh = (cam_data["width"], cam_data["height"])
    calibration_cam = data.get(
        "calibrated_sensor", cam_data["calibrated_sensor_token"]
    )
    labels: Optional[List[Label]] = None
    if boxes is not None:
        labels = parse_labels(
            data, boxes, ego_pose_cam, calibration_cam, img_wh
        )

    frame = Frame(
        name=os.path.basename(cam_filepath),
        videoName=scene_name,
        frameIndex=frame_index,
        url=cam_filepath,
        timestamp=cam_data["timestamp"],
        extrinsics=get_extrinsics(ego_pose_cam, calibration_cam),
        intrinsics=calibration_to_intrinsics(calibration_cam),
        size=ImageSize(width=img_wh[0], height=img_wh[1]),
        labels=labels,
    )
    next_token: Optional[str] = None
    if (
        cam_data["next"] != ""
        and not data.get("sample_data", cam_data["next"])["is_key_frame"]
    ):
        next_token = cam_data["next"]
    return frame, next_token


def parse_sequence(
    data: NuScenes, add_nonkey_frames: bool, scene_info: Tuple[str, str]
) -> Tuple[List[Frame], List[FrameGroup]]:
    """Parse a full NuScenes sequence and convert it into scalabel frames."""
    sample_token, scene_name = scene_info
    frames, groups = [], []
    frame_index = 0
    while sample_token:
        sample = data.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = data.get("sample_data", lidar_token)
        calibration_lidar = data.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )
        timestamp = lidar_data["timestamp"]
        lidar_filepath = lidar_data["filename"]
        ego_pose = data.get("ego_pose", lidar_data["ego_pose_token"])
        frame_names = []
        next_nonkey_frames = []
        for cam in cams:
            cam_token = sample["data"][cam]
            boxes = data.get_boxes(lidar_token)
            frame, next_token = parse_frame(
                data, scene_name, frame_index, cam_token, boxes
            )
            frame_names.append(frame.name)
            frames.append(frame)
            if add_nonkey_frames and next_token is not None:
                next_nonkey_frames.append(next_token)

        group = FrameGroup(
            name=sample_token,
            videoName=scene_name,
            frameIndex=frame_index,
            url=lidar_filepath,
            extrinsics=get_extrinsics(ego_pose, calibration_lidar),
            timestamp=timestamp,
            frames=frame_names,
            labels=parse_labels(
                data,
                data.get_boxes(lidar_token),
                ego_pose,
                calibration_lidar,
                in_image_frame=False,
            ),
        )
        groups.append(group)
        frame_index += 1

        nonkey_count = 0
        while len(next_nonkey_frames) > 0:
            new_next_nonkey_frames = []
            frame_names = []
            nonkey_count += 1
            for cam_token in next_nonkey_frames:
                assert cam_token is not None, "camera for non-key missing!"
                frame, next_token = parse_frame(
                    data, scene_name, frame_index, cam_token
                )
                frame_names.append(frame.name)
                frames.append(frame)
                if add_nonkey_frames and next_token is not None:
                    new_next_nonkey_frames.append(next_token)

            group = FrameGroup(
                name=sample_token + f"_{nonkey_count}",
                videoName=scene_name,
                frameIndex=frame_index,
                frames=frame_names,
            )
            groups.append(group)
            next_nonkey_frames = new_next_nonkey_frames
            frame_index += 1

        sample_token = sample["next"]

    return frames, groups


def from_nuscenes(
    data_path: str,
    version: str,
    split: str,
    nproc: int = NPROC,
    add_nonkey_frames: bool = False,
) -> Dataset:
    """Convert NuScenes dataset to Scalabel format."""
    data, df = load_data(data_path, version)
    scene_names_per_split = create_splits_scenes()

    first_sample_tokens = []
    for token, name in zip(df.first_sample_token.values, df.scene_name.values):
        if name in scene_names_per_split[split]:
            first_sample_tokens.append(token)

    func = partial(parse_sequence, data, add_nonkey_frames)
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
    assert NuScenes is not None, (
        "Please install the requirements in scripts/optional.txt to use"
        "NuScenes conversion."
    )

    if "mini" in args.version:
        splits_to_iterate = ["mini_train", "mini_val"]
    elif "test" in args.version:
        splits_to_iterate = ["test"]
    else:
        splits_to_iterate = ["train", "val"]

    if len(args.splits) > 0:
        assert all(
            (s in splits_to_iterate for s in args.splits)
        ), f"Invalid splits, please select splits from {splits_to_iterate}!"
        splits_to_iterate = args.splits

    if args.output is not None:
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        out_dir = args.output
    else:
        out_dir = args.input

    for split in splits_to_iterate:
        result = from_nuscenes(
            args.input,
            args.version,
            split,
            args.nproc,
            args.add_non_key,
        )
        save(os.path.join(out_dir, f"scalabel_{split}.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
