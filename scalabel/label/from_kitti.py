"""Convert kitti to Scalabel format."""
import argparse
import os
import os.path as osp
from typing import List

import pdb
import cv2
from scipy.spatial.transform import Rotation as R

from ..common.parallel import NPROC
from .kitti_utlis import (
    read_calib,
    read_calib_det,
    read_oxts,
    KittiPoseParser,
    list_from_file,
    cameratoimage,
)

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

# 396, 1333
val_sets = ["0001", "0004", "0011", "0012", "0013", "0014", "0015", "0018"]
mini_sets = ["0001"]


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
        "--mode",
        default="mini",
        choices=["mini", "subtrain", "subval", "train", "test"],
        help="mode for kitti dataset",
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


def parse_labels(label_file, adjust_center, trackid_maps, global_track_id):
    anns = dict()
    labels = list_from_file(label_file)

    for label in labels:
        label = label.split()
        if not int(label[0]) in anns.keys():
            anns[int(label[0])] = []
        cat = label[2]
        if cat in ["DontCare"]:
            continue

        class_name = kitti_cats[cat]

        if label[1] in trackid_maps.keys():
            track_id = trackid_maps[label[1]]
        else:
            track_id = global_track_id
            trackid_maps[label[1]] = track_id
            global_track_id += 1

        x1, y1, x2, y2 = (
            float(label[6]),
            float(label[7]),
            float(label[8]),
            float(label[9]),
        )

        if adjust_center:
            # KITTI GT uses the bottom of the car as center (x, 0, z).
            # Prediction uses center of the bbox as center (x, y, z).
            # So we align them to the bottom center as GT does
            y_cen_adjust = float(label[10]) / 2.0
        else:
            y_cen_adjust = 0.0

        box3d = Box3D(
            orientation=(0.0, float(label[16]), 0.0),
            location=(
                float(label[13]),
                float(label[14]) - y_cen_adjust,
                float(label[15]),
            ),
            dimension=(float(label[10]), float(label[11]), float(label[12])),
            alpha=float(label[5]),
        )

        box2d = Box2D(x1=x1, y1=y1, x2=x2, y2=y2)

        anns[int(label[0])].append(
            Label(
                category=class_name,
                box2d=box2d,
                box3d=box3d,
                id=str(track_id),
            )
        )

    return anns, trackid_maps, global_track_id


def from_kitti_det(
    data_dir, mode, det_val_sets, adjust_center=True
) -> List[Frame]:
    frames = []

    img_dir = osp.join(data_dir, "image_2")
    label_dir = osp.join(data_dir, "label_2")
    cali_dir = osp.join(data_dir, "calib")

    if not osp.exists(img_dir):
        print(f"Folder {img_dir} is not found")
        return None

    if not osp.exists(label_dir):
        label_dir = None

    img_names = sorted(os.listdir(img_dir))

    rotation = (0, 0, 0)
    position = (0, 0, 0)

    cam2global = Extrinsics(location=position, rotation=rotation)

    for img_id, img_name in enumerate(img_names):
        if mode == "subtrain":
            if osp.splitext(img_name)[0] in det_val_sets:
                continue
        elif mode == "subval":
            if osp.splitext(img_name)[0] not in det_val_sets:
                continue
        print(f"DET: {img_name.split('.')[0]}")
        img = cv2.imread(osp.join(img_dir, img_name))

        height, width, _ = img.shape
        image_size = ImageSize(height=height, width=width)

        projection = read_calib_det(cali_dir, int(img_name.split(".")[0]))

        intrinsics = Intrinsics(
            focal=(projection[0][0], projection[1][1]),
            center=(projection[0][2], projection[1][2]),
        )

        labels = []
        if label_dir:
            track_id = 0
            label_file = osp.join(
                label_dir, "{}.txt".format(img_name.split(".")[0])
            )
            label_det = list_from_file(label_file)
            for label in label_det:
                label = label.split()
                cat = label[0]
                if cat in ["DontCare"]:
                    continue

                class_name = kitti_cats[cat]

                x1, y1, x2, y2 = (
                    float(label[4]),
                    float(label[5]),
                    float(label[6]),
                    float(label[7]),
                )

                if adjust_center:
                    # KITTI GT uses the bottom of the car as center (x, 0, z).
                    # Prediction uses center of the bbox as center (x, y, z).
                    # So we align them to the bottom center as GT does
                    y_cen_adjust = float(label[8]) / 2.0
                else:
                    y_cen_adjust = 0.0

                box3d = Box3D(
                    orientation=(0.0, float(label[14]), 0.0),
                    location=(
                        float(label[11]),
                        float(label[12]) - y_cen_adjust,
                        float(label[13]),
                    ),
                    dimension=(
                        float(label[8]),
                        float(label[9]),
                        float(label[10]),
                    ),
                    alpha=float(label[3]),
                )

                box2d = Box2D(x1=x1, y1=y1, x2=x2, y2=y2)

                labels.append(
                    Label(
                        category=class_name,
                        box2d=box2d,
                        box3d=box3d,
                        id=str(track_id),
                    )
                )
                track_id += 1

        f = Frame(
            name=img_name,
            frame_index=img_id,
            size=image_size,
            extrinsics=cam2global,
            intrinsics=intrinsics,
            labels=labels,
        )
        frames.append(f)

    return frames


def from_kitti(
    data_dir, data_type, mode, det_val_sets, adjust_center=True
) -> List[Frame]:
    """Function converting kitti data to Scalabel format."""
    if data_type == "detection":
        return from_kitti_det(data_dir, mode, det_val_sets)

    frames = []

    img_dir = osp.join(data_dir, "image_02")
    label_dir = osp.join(data_dir, "label_02")
    cali_dir = osp.join(data_dir, "calib")
    oxt_dir = osp.join(data_dir, "oxts")

    if not osp.exists(img_dir):
        print(f"Folder {img_dir} is not found")
        return None

    if not osp.exists(label_dir):
        label_dir = None

    vid_names = sorted(os.listdir(img_dir))

    global_track_id = 0

    for vid_id, vid_name in enumerate(vid_names):
        if mode == "subtrain":
            if vid_name in val_sets:
                continue
        elif mode == "subval":
            if vid_name not in val_sets:
                continue
        elif mode == "mini":
            if vid_name not in mini_sets:
                continue
        print(f"VID: {vid_name}")

        trackid_maps = dict()
        ind2id = dict()

        img_names = sorted(
            [
                f.path
                for f in os.scandir(osp.join(img_dir, vid_name))
                if f.is_file() and f.name.endswith("png")
            ]
        )

        projection = read_calib(cali_dir, int(vid_name))

        if label_dir:
            label_file = osp.join(label_dir, "{}.txt".format(vid_name))
            labels_dict, trackid_maps, global_track_id = parse_labels(
                label_file, adjust_center, trackid_maps, global_track_id
            )

        for fr, img_name in enumerate(sorted(img_names)):
            if mode == "mini" and fr == 2:
                break
            img = cv2.imread(img_name)

            height, width, _ = img.shape
            image_size = ImageSize(height=height, width=width)

            fields = read_oxts(oxt_dir, int(vid_name))
            poses = [KittiPoseParser(fields[i]) for i in range(len(fields))]

            rotation = tuple(
                R.from_matrix(poses[fr].rotation).as_euler("xyz").tolist()
            )
            position = tuple((poses[fr].position - poses[0].position).tolist())

            cam2global = Extrinsics(location=position, rotation=rotation)

            intrinsics = Intrinsics(
                focal=(projection[0][0], projection[1][1]),
                center=(projection[0][2], projection[1][2]),
            )

            if label_dir:
                if not fr in labels_dict.keys():
                    labels = []
                else:
                    labels = labels_dict[fr]
            else:
                labels = []

            f = Frame(
                name=img_name,
                video_name=vid_name,
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

    if args.mode == "test":
        subset = "testing"
        mode = None
    else:
        subset = "training"
        if args.mode == "train":
            mode = None
        else:
            mode = args.mode

    output_name = f"{args.data_type}_{args.mode}.json"

    if args.data_type == "detection":
        with open(
            osp.join(args.input_dir, "detection/detection_val.txt"), "r"
        ) as f:
            det_val_sets = f.read().splitlines()
    else:
        det_val_sets = []

    data_dir = osp.join(args.input_dir, args.data_type, subset)

    scalabel = from_kitti(
        data_dir,
        data_type=args.data_type,
        mode=mode,
        det_val_sets=det_val_sets,
    )

    save(
        osp.join(args.output_dir, output_name),
        scalabel,
        args.nproc,
    )


if __name__ == "__main__":
    run(parse_arguments())
