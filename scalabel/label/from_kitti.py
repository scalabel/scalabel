"""Convert kitti to Scalabel format."""
import argparse
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from ..common.parallel import NPROC
from .io import save
from .kitti_utlis import (
    KittiPoseParser,
    list_from_file,
    read_calib,
    read_calib_det,
    read_oxts,
)
from .typing import (
    Box2D,
    Box3D,
    Extrinsics,
    Frame,
    ImageSize,
    Intrinsics,
    Label,
)

kitti_cats = {
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist",
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Tram": "tram",
    "Person": "pedestrian",
    "Person_sitting": "pedestrian",
    "Misc": "misc",
    "DontCare": "dontcare",
}


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to scalabel")
    parser.add_argument(
        "--input_dir",
        "-i",
        help="path to the input coco label file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="path to save scalabel format label file",
    )
    parser.add_argument(
        "--split",
        default="training",
        choices=["training", "testing"],
        help="split for kitti dataset",
    )
    parser.add_argument(
        "--data_type",
        default="tracking",
        choices=["tracking", "detection"],
        help="type of kitti dataset",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    return parser.parse_args()


def parse_label(
    data_type: str,
    label_file: str,
    trackid_maps: Dict[str, int],
    global_track_id: int,
) -> Tuple[Dict[int, List[Label]], Dict[str, int], int]:
    """Function parsing tracking / detection labels."""
    if data_type == "tracking":
        offset = 2
    else:
        offset = 0

    labels_dict: Dict[int, List[Label]] = dict()

    labels = list_from_file(label_file)
    track_id = -1

    for label_line in labels:
        label = label_line.split()

        if data_type == "tracking":
            seq_id = int(label[0])
        else:
            seq_id = 0

        if seq_id not in labels_dict.keys():
            labels_dict[seq_id] = []

        cat = label[0 + offset]
        if cat in ["DontCare"]:
            continue
        class_name = kitti_cats[cat]

        if data_type == "tracking":
            if label[1] in trackid_maps.keys():
                track_id = trackid_maps[label[1]]
            else:
                track_id = global_track_id
                trackid_maps[label[1]] = track_id
                global_track_id += 1
        else:
            track_id += 1

        x1, y1, x2, y2 = (
            float(label[4 + offset]),
            float(label[5 + offset]),
            float(label[6 + offset]),
            float(label[7 + offset]),
        )

        y_cen_adjust = float(label[8 + offset]) / 2.0

        box3d = Box3D(
            orientation=(0.0, float(label[14 + offset]), 0.0),
            location=(
                float(label[11 + offset]),
                float(label[12 + offset]) - y_cen_adjust,
                float(label[13 + offset]),
            ),
            dimension=(
                float(label[8 + offset]),
                float(label[9 + offset]),
                float(label[10 + offset]),
            ),
            alpha=float(label[3 + offset]),
        )

        box2d = Box2D(x1=x1, y1=y1, x2=x2, y2=y2)

        labels_dict[seq_id].append(
            Label(
                category=class_name,
                box2d=box2d,
                box3d=box3d,
                id=str(track_id),
            )
        )

    return labels_dict, trackid_maps, global_track_id


def from_kitti_det(
    data_dir: str,
    data_type: str,
) -> List[Frame]:
    """Function converting kitti detection data to Scalabel format."""
    frames = []

    img_dir = osp.join(data_dir, "image_2")
    label_dir = osp.join(data_dir, "label_2")
    cali_dir = osp.join(data_dir, "calib")

    assert osp.exists(img_dir), f"Folder {img_dir} is not found"

    img_names = sorted(os.listdir(img_dir))

    global_track_id = 0

    rotation = (0, 0, 0)
    position = (0, 0, 0)

    cam2global = Extrinsics(location=position, rotation=rotation)

    for img_id, img_name in enumerate(img_names):
        trackid_maps: Dict[str, int] = dict()

        with Image.open(osp.join(img_dir, img_name)) as img:
            width, height = img.size
            image_size = ImageSize(height=height, width=width)

        projection = read_calib_det(cali_dir, int(img_name.split(".")[0]))

        intrinsics = Intrinsics(
            focal=(projection[0][0], projection[1][1]),
            center=(projection[0][2], projection[1][2]),
        )

        if osp.exists(label_dir):
            label_file = osp.join(
                label_dir, "{}.txt".format(img_name.split(".")[0])
            )
            labels_dict, _, _ = parse_label(
                data_type, label_file, trackid_maps, global_track_id
            )

            labels = labels_dict[0]
        else:
            labels = []

        image_name = osp.join(img_dir, img_name)
        image_name = data_type + image_name.split(data_type)[-1]

        video_name = "/".join(image_name.split("/")[:-1])

        f = Frame(
            name=img_name,
            video_name=video_name,
            frame_index=img_id,
            size=image_size,
            extrinsics=cam2global,
            intrinsics=intrinsics,
            labels=labels,
        )
        frames.append(f)

    return frames


def from_kitti(
    data_dir: str,
    data_type: str,
) -> List[Frame]:
    """Function converting kitti data to Scalabel format."""
    if data_type == "detection":
        return from_kitti_det(data_dir, data_type)

    frames = []

    img_dir = osp.join(data_dir, "image_02")
    label_dir = osp.join(data_dir, "label_02")
    cali_dir = osp.join(data_dir, "calib")
    oxt_dir = osp.join(data_dir, "oxts")

    assert osp.exists(img_dir), f"Folder {img_dir} is not found"

    vid_names = sorted(os.listdir(img_dir))

    global_track_id = 0

    for vid_name in vid_names:
        trackid_maps: Dict[str, int] = dict()

        img_names = sorted(
            [
                f.path
                for f in os.scandir(osp.join(img_dir, vid_name))
                if f.is_file() and f.name.endswith("png")
            ]
        )

        projection = read_calib(cali_dir, int(vid_name))

        if osp.exists(label_dir):
            label_file = osp.join(label_dir, "{}.txt".format(vid_name))
            labels_dict, trackid_maps, global_track_id = parse_label(
                data_type, label_file, trackid_maps, global_track_id
            )

        for fr, img_name in enumerate(sorted(img_names)):
            with Image.open(img_name) as img:
                width, height = img.size
                image_size = ImageSize(height=height, width=width)

            fields = read_oxts(oxt_dir, int(vid_name))
            poses = [KittiPoseParser(fields[i]) for i in range(len(fields))]

            rotation = tuple(
                R.from_matrix(poses[fr].rotation).as_euler("xyz").tolist()
            )
            position = tuple(
                np.array(poses[fr].position - poses[0].position).tolist()
            )

            cam2global = Extrinsics(location=position, rotation=rotation)

            intrinsics = Intrinsics(
                focal=(projection[0][0], projection[1][1]),
                center=(projection[0][2], projection[1][2]),
            )

            if osp.exists(label_dir):
                if not fr in labels_dict.keys():
                    labels = []
                else:
                    labels = labels_dict[fr]
            else:
                labels = []

            img_name = data_type + img_name.split(data_type)[-1]

            video_name = "/".join(img_name.split("/")[:-1])

            f = Frame(
                name=img_name.split("/")[-1],
                video_name=video_name,
                frame_index=fr,
                size=image_size,
                extrinsics=cam2global,
                intrinsics=intrinsics,
                labels=labels,
            )
            frames.append(f)

    return frames


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_name = f"{args.data_type}_{args.split}.json"

    data_dir = osp.join(args.input_dir, args.data_type, args.split)

    scalabel = from_kitti(
        data_dir,
        data_type=args.data_type,
    )

    save(
        osp.join(args.output_dir, output_name),
        scalabel,
        args.nproc,
    )


if __name__ == "__main__":
    run(parse_arguments())
