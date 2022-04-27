"""Conversion script from Scalabel to NuScenes format."""
import argparse
import json
import operator
from typing import Dict

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

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
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


def to_nuscenes(
    dataset: Dataset, mode: str, metadata: Dict[str, bool]
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

                tracking_id = label.id
                if prev_loc.get(tracking_id) is not None:
                    velocity = translation - prev_loc[tracking_id]
                else:
                    velocity = [0.0, 0.0, 0.0]
                prev_loc[tracking_id] = translation

                if mode == "detection":
                    nusc_anno = {
                        "sample_token": token,
                        "translation": translation.tolist(),
                        "size": dimension,
                        "rotation": rotation.elements.tolist(),
                        "velocity": [velocity[0], velocity[1]],
                        "detection_name": label.category,
                        "detection_score": label.score,
                        "attribute_name": max(
                            cls_attr_dist[label.category].items(),
                            key=operator.itemgetter(1),
                        )[0],
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
