"""Detecting ground in point clouds.

Script for detecting ground in point clouds and writing it back to the PLY file
"""

import argparse
import json
import os
import sys
import urllib.request
from typing import List, Tuple, Union

import numpy as np
import plyfile
import yaml
from tqdm import tqdm

# Define the ground return type
GroundOutputType = Union[
    None, Tuple[np.ndarray, float, np.ndarray, np.ndarray]
]


def estimate_ground_plane(
    points: np.ndarray,
    sample_size: float,
    iters: int,
    dist_cutoff: List[float],
    height_cutoff: List[float],
    expected_normal: np.ndarray,
    max_normal_deviation: float,
    inlier_cutoff: float = 0.15,
) -> GroundOutputType:
    """Detect ground plane in the point cloud using RANSAC.

    Detect ground in points by using RANSAC to find largest plane
    in point cloud that has normal close to what is expected
    """
    assert 0 < sample_size < 1
    assert len(dist_cutoff) == 2
    assert len(height_cutoff) == 2

    depths = np.linalg.norm(points, axis=-1)
    height_filtered = np.logical_and(
        points[:, 2] > height_cutoff[0], points[:, 2] < height_cutoff[1]
    )
    dist_filtered = np.logical_and(
        depths < dist_cutoff[1], depths > dist_cutoff[0]
    )
    valid_indices = np.logical_and(height_filtered, dist_filtered)
    valid_points = points[valid_indices]
    point_indices = np.arange(valid_points.shape[0])

    num_samples = int(round(sample_size * valid_points.shape[0]))

    max_inliers = 0
    best_plane = None
    best_plane_inliers = None
    best_plane_outliers = None

    for _ in range(iters):
        sample_indices = np.random.choice(
            point_indices, size=num_samples, replace=False
        )
        sample_points = valid_points[sample_indices]

        # Use PCA to estimate normal
        covariance = np.cov(sample_points, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(covariance)
        sample_normal = eigvecs[:, np.argmin(eigvals)]

        if (
            np.arccos(sample_normal.dot(expected_normal))
            > max_normal_deviation
        ):
            continue

        sample_center = np.average(sample_points, axis=0)
        diffs = valid_points - sample_center
        dists_to_plane = (diffs).dot(sample_normal)

        same_plane_indices = np.abs(dists_to_plane) < inlier_cutoff
        same_plane_points = valid_points[same_plane_indices]

        if same_plane_points.shape[0] > max_inliers:
            max_inliers = same_plane_points.shape[0]
            offset = -sample_normal.dot(sample_center)

            best_plane = (sample_normal, offset)
            best_plane_inliers = same_plane_points
            best_plane_outliers = valid_points[
                np.logical_not(same_plane_indices)
            ]

    if best_plane:
        return (
            np.array(best_plane[0]),
            float(best_plane[1]),
            np.array(best_plane_inliers),
            np.array(best_plane_outliers),
        )
    return None


def main() -> None:
    """Run."""
    parser = argparse.ArgumentParser(
        description="Find ground plane and write it to PLY file"
    )
    parser.add_argument(
        "--bdd_items", help="Input BDD Data json / yaml file", required=True
    )
    parser.add_argument(
        "--output", help="Output BDD Data json / yaml file", required=True
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations to run for RANSAC.",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--sample_size",
        help="Fraction of points to use as sample",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--min_dist",
        help="Minimum distance of points from "
        "origin to be considered valid",
        default=3.0,
        type=float,
    )
    parser.add_argument(
        "--max_dist",
        help="Maximum distance of points from "
        "origin to be considered valid",
        default=25.0,
        type=float,
    )
    parser.add_argument(
        "--min_height",
        help="Minimum height of points, value "
        "on z-axis to be considered valid",
        default=-2.0,
        type=float,
    )
    parser.add_argument(
        "--max_height",
        help="Maximum height of points, value "
        "on z-axis to be considered valid",
        default=-1.0,
        type=float,
    )
    parser.add_argument(
        "--expected_normal",
        help="Expected normal of the ground plane",
        nargs="+",
        default=[0, 0, 1],
        type=float,
    )
    parser.add_argument(
        "--max_normal_deviation",
        help="Maximum deviation from expected normal " "in radians",
        default=0.15,
        type=float,
    )
    parser.add_argument(
        "--tracking",
        help="Set flag if tracking. "
        "Will make all ground planes have same id.",
        action="store_true",
    )

    args = parser.parse_args()

    if len(args.expected_normal) != 3:
        print("Expected normal must be a 3 element array.")
        sys.exit(0)

    expected_normal = np.array(args.expected_normal)

    # Check file extensions
    supported_ext = [".yaml", ".json"]
    if not os.path.splitext(args.bdd_items)[-1] in supported_ext:
        raise ValueError(
            "[Error] Argument --bdd_items should be either"
            + " a json or a yaml file."
        )
    if not os.path.splitext(args.output)[-1] in supported_ext:
        raise ValueError(
            "[Error] Argument --output should be either"
            + " a json or a yaml file."
        )

    with open(args.bdd_items, "r") as bdd_file:
        if os.path.splitext(args.bdd_items)[-1] == ".json":
            bdd_data = json.load(bdd_file)
        else:
            bdd_data = yaml.safe_load(bdd_file)

    if not isinstance(bdd_data, list):
        raise ValueError(
            "[Error] Contents in the json file are not in"
            + " valid label format."
        )

    for item in bdd_data:
        if "labels" not in item.keys():
            item["labels"] = []

    label_ids = [label["id"] for item in bdd_data for label in item["labels"]]
    max_label_id = max(label_ids) if label_ids else 0

    for item in tqdm(bdd_data, ascii=True):
        url = item["url"]
        with urllib.request.urlopen(url) as http_response:
            ply_data = plyfile.PlyData.read(http_response)
            points = np.concatenate(
                [
                    ply_data["vertex"]["x"][..., None],
                    ply_data["vertex"]["y"][..., None],
                    ply_data["vertex"]["z"][..., None],
                ],
                axis=-1,
            )

        results = estimate_ground_plane(
            points,
            args.sample_size,
            args.iterations,
            [args.min_dist, args.max_dist],
            [args.min_height, args.max_height],
            expected_normal,
            args.max_normal_deviation,
        )

        if not results:
            continue

        normal, offset = results[:2]

        plane_label = {
            "id": max_label_id + 1,
            "category": "",
            "attributes": {},
            "manualShape": False,
            "box2d": None,
            "poly2d": None,
            "box3d": None,
            "plane3d": {
                "center": {"x": 0, "y": 0, "z": -float(offset)},
                "orientation": {
                    "x": float(normal[0]),
                    "y": float(normal[1]),
                    "z": float(normal[2]),
                },
            },
            "customs": {},
        }

        item["labels"].append(plane_label)

        if not args.tracking:
            max_label_id += 1

    with open(args.output, "w") as output_file:
        if os.path.splitext(args.output)[-1] == ".json":
            json.dump(bdd_data, output_file)
        else:
            yaml.safe_dump(bdd_data, output_file)


if __name__ == "__main__":
    main()
