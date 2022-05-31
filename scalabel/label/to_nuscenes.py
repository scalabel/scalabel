"""Conversion script from Scalabel to NuScenes format."""
import argparse
import json
import operator
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..common.io import open_write_text
from ..common.parallel import NPROC
from ..common.typing import DictStrAny, NDArrayF64
from .io import load
from .typing import Dataset

try:
    from nuscenes import NuScenes
    from nuscenes.eval.detection.constants import DETECTION_NAMES
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Quaternion
except ImportError:
    NuScenes = None

DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}

tracking_cats = [
    "bicycle",
    "motorcycle",
    "pedestrian",
    "bus",
    "car",
    "trailer",
    "truck",
]


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Scalabel to nuScenes format")
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
        help="path to save nuscenes formatted label file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="tracking",
        choices=["tracking", "detection"],
        help="conversion mode: detection or tracking.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        default=[],
        help="Modalities / Data used: camera, lidar, radar, map, external",
    )
    return parser.parse_args()


def get_attributes(name: str, velocity: List[float]) -> str:
    """Get nuScenes attributes."""
    if np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) > 0.2:
        if name in [
            "car",
            "construction_vehicle",
            "bus",
            "truck",
            "trailer",
        ]:
            attr = "vehicle.moving"
        elif name in ["bicycle", "motorcycle"]:
            attr = "cycle.with_rider"
        else:
            attr = DefaultAttribute[name]
    else:
        if name in ["pedestrian"]:
            attr = "pedestrian.standing"
        elif name in ["bus"]:
            attr = "vehicle.stopped"
        else:
            attr = DefaultAttribute[name]
    return attr


def to_nuscenes(
    dataset: Dataset,
    mode: str,
    metadata: Dict[str, bool],
    attr_by_velocity: bool = True,
) -> DictStrAny:
    """Conver Scalabel format prediction into nuScenes JSON file."""
    results: DictStrAny = {}

    assert dataset.frames is not None
    key_frames = [f for f in dataset.frames if not "_" in f.name]

    if mode == "detection":
        categories = DETECTION_NAMES
    else:
        categories = tracking_cats

    for frame in key_frames:
        annos = []
        token = frame.name

        if frame.frameIndex == 0:
            prev_loc: Dict[str, NDArrayF64] = {}
        if frame.labels is not None:
            for label in frame.labels:
                assert label.category is not None
                if not label.category in categories:
                    continue

                assert frame.extrinsics is not None
                sensor2global = R.from_euler(
                    "xyz", frame.extrinsics.rotation
                ).as_matrix()

                assert label.box3d is not None
                translation = np.dot(
                    sensor2global, label.box3d.location
                ) + np.array(frame.extrinsics.location)

                # Using extrinsic rotation here to align with Pytorch3D
                x, y, z, w = R.from_euler(
                    "XYZ", label.box3d.orientation
                ).as_quat()
                quat = Quaternion([w, x, y, z])
                x, y, z, w = R.from_matrix(sensor2global).as_quat()
                rotation = Quaternion([w, x, y, z]) * quat

                h, w, l = label.box3d.dimension
                dimension = [w, l, h]
                if mode == "detection":
                    dimension = [d if d >= 0 else 0.1 for d in dimension]

                velocity = np.dot(
                    sensor2global, np.array(label.box3d.velocity)
                ).tolist()

                if attr_by_velocity:
                    attribute_name = get_attributes(label.category, velocity)
                else:
                    attribute_name = DefaultAttribute[label.category]

                tracking_id = label.id
                if mode == "detection":
                    nusc_anno = {
                        "sample_token": token,
                        "translation": translation.tolist(),
                        "size": dimension,
                        "rotation": rotation.elements.tolist(),
                        "velocity": [velocity[0], velocity[1]],
                        "detection_name": label.category,
                        "detection_score": label.score,
                        "attribute_name": attribute_name,
                    }
                else:
                    nusc_anno = {
                        "sample_token": token,
                        "translation": translation.tolist(),
                        "size": dimension,
                        "rotation": rotation.elements.tolist(),
                        "velocity": [velocity[0], velocity[1]],
                        "tracking_id": tracking_id,
                        "tracking_name": label.category,
                        "tracking_score": label.score,
                    }
                annos.append(nusc_anno)
        results[token] = annos

    nusc_annos = {
        "results": results,
        "meta": metadata,
    }

    return nusc_annos


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    assert NuScenes is not None, (
        "Please install the requirements in scripts/optional.txt to use"
        "NuScenes conversion."
    )
    metadata = {
        "use_camera": False,
        "use_lidar": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    assert len(args.metadata) > 0, "Please state the used modality and data!"

    assert all(("use_" + m in metadata for m in args.metadata)), (
        "Invalid metadata, please select from "
        f"{[m.replace('use_', '') for m in list(metadata.keys())]}!"
    )

    for m in args.metadata:
        metadata["use_" + m] = True

    dataset = load(args.input, args.nproc)

    nusc = to_nuscenes(dataset, args.mode, metadata)

    with open_write_text(args.output) as f:
        json.dump(nusc, f)


if __name__ == "__main__":
    run(parse_arguments())
