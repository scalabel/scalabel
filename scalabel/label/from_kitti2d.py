"""Convert KITTI 2D to Scalabel format."""
import argparse
import copy
import os
import os.path as osp
from typing import Dict, List, Tuple

from PIL import Image
from scipy.spatial.transform import Rotation as R

from ..common.parallel import NPROC
from .transforms import xyxy_to_box2d
from .io import save
from .kitti_utlis import list_from_file
from .typing import (
    Box3D,
    Category,
    Config,
    Dataset,
    Frame,
    ImageSize,
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

kitti_used_cats = ["pedestrian", "cyclist", "car", "truck", "tram", "misc"]


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="coco to scalabel")
    parser.add_argument(
        "--input-dir",
        "-i",
        help="path to the input coco label file",
    )
    parser.add_argument(
        "--output-dir",
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
        "--data-type",
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

    labels_dict: Dict[int, List[Label]] = {}

    labels = list_from_file(label_file)
    track_id = -1

    for label_line in labels:
        label = label_line.split()

        if data_type == "tracking":
            seq_id = int(label[0])
        else:
            seq_id = 0

        if seq_id not in labels_dict:
            labels_dict[seq_id] = []

        cat = label[0 + offset]
        # if cat in ["DontCare"]:
        #     continue
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

        box2d = xyxy_to_box2d(
            float(label[4 + offset]),
            float(label[5 + offset]),
            float(label[6 + offset]),
            float(label[7 + offset]),
        )

        labels_dict[seq_id].append(
            Label(category=class_name, box2d=box2d, id=str(track_id))
        )

    return labels_dict, trackid_maps, global_track_id


def from_kitti_det(data_dir: str, data_type: str) -> Dataset:
    """Function converting kitti detection data to Scalabel format."""
    frames = []

    label_dir = osp.join(data_dir, "label_2")

    img_names = sorted(os.listdir(label_dir))

    global_track_id = 0
    for frame_idx, velodyne_name in enumerate(img_names):
        img_name = velodyne_name.split(".")[0] + ".png"
        trackid_maps: Dict[str, int] = {}

        cam = "image_2"
        img_dir = osp.join(data_dir, cam)
        with Image.open(osp.join(img_dir, img_name)) as img:
            width, height = img.size
            image_size = ImageSize(height=height, width=width)

        if osp.exists(label_dir):
            label_file = osp.join(
                label_dir, f"{img_name.split('.')[0]}.txt"
            )
            labels_dict, _, _ = parse_label(
                data_type, label_file, trackid_maps, global_track_id
            )
            labels = labels_dict[0]
        else:
            labels = []

        f = Frame(
            name=img_name,
            frameIndex=frame_idx,
            url=img_name,
            size=image_size,
            labels=labels,
        )
        frames.append(f)

    cfg = Config(categories=[Category(name=n) for n in kitti_used_cats])
    dataset = Dataset(frames=frames, config=cfg)
    return dataset


def from_kitti(data_dir: str, data_type: str) -> Dataset:
    """Function converting kitti data to Scalabel format."""
    if data_type == "detection":
        return from_kitti_det(data_dir, data_type)

    frames = []
    cam = "image_02"

    video_dir = osp.join(data_dir, cam)
    label_dir = osp.join(data_dir, "label_02")

    video_names = sorted(os.listdir(video_dir))

    global_track_id = 0
    total_img_names = {}

    for video_name in video_names:
        img_dir = osp.join(data_dir, cam)
        total_img_names[cam] = sorted(
            [
                f.path
                for f in os.scandir(osp.join(img_dir, video_name))
                if f.is_file() and f.name.endswith("png")
            ]
        )
        trackid_maps: Dict[str, int] = {}

        if osp.exists(label_dir):
            label_file = osp.join(label_dir, f"{video_name}.txt")
            labels_dict, trackid_maps, global_track_id = parse_label(
                data_type, label_file, trackid_maps, global_track_id
            )

        for frame_idx in range(len(total_img_names[cam])):
            frame_names = []
            for cam, img_names in total_img_names.items():
                img_name = img_names[frame_idx]
                with Image.open(img_name) as img:
                    width, height = img.size
                    image_size = ImageSize(height=height, width=width)

                if osp.exists(label_dir):
                    if not frame_idx in labels_dict:
                        labels = []
                    else:
                        labels = labels_dict[frame_idx]
                else:
                    labels = []

                img_name_list = img_name.split(f"{cam}/")[-1].split("/")
                vid_name, img_name = img_name_list

                frame = Frame(
                    name=img_name,
                    videoName=vid_name,
                    frameIndex=frame_idx,
                    url="/".join(img_name_list),
                    size=image_size,
                    labels=labels,
                )
                frame_names.append(img_name)
                frames.append(frame)

    cfg = Config(categories=[Category(name=n) for n in kitti_used_cats])
    dataset = Dataset(frames=frames, config=cfg)
    return dataset


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = f"{args.data_type}_{args.split}.json"

    data_dir = osp.join(args.input_dir, args.data_type, args.split)

    scalabel = from_kitti(data_dir, data_type=args.data_type)

    save(osp.join(args.output_dir, output_name), scalabel)


if __name__ == "__main__":
    run(parse_arguments())
