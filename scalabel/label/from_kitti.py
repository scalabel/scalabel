"""Convert kitti to Scalabel format."""
import argparse
import copy
import math
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from scalabel.label.utils import cart2hom, rotation_y_to_alpha

from ..common.parallel import NPROC
from ..common.typing import NDArrayF64
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


def heading_transform(box3d: Box3D, calib: NDArrayF64) -> float:
    """Transform heading from camera coordinate into lidar system."""
    rot_y = box3d.orientation[1]
    tm: NDArrayF64 = calib @ get_box_transformation_matrix(
        box3d.location,
        (box3d.dimension[0], box3d.dimension[1], box3d.dimension[2]),
        rot_y,
    )
    pt1 = np.array([-0.5, 0.5, 0, 1.0])
    pt2 = np.array([0.5, 0.5, 0, 1.0])
    pt1 = np.matmul(tm, pt1).tolist()
    pt2 = np.matmul(tm, pt2).tolist()
    return -math.atan2(pt2[2] - pt1[2], pt2[0] - pt1[0])


def parse_lidar_labels(
    labels: List[Label],
    calib: Extrinsics,
) -> List[Label]:
    """Function parsing tracking / detection labels into lidar frames."""
    lidar2cam_mat = get_matrix_from_extrinsics(calib)
    cam2lidar_mat = np.linalg.inv(lidar2cam_mat)

    new_labels = []
    for label in labels:
        if label.box3d is not None:
            box3d = label.box3d

            center = np.array([box3d.location])
            center_lidar = tuple(
                np.dot(cam2lidar_mat, cart2hom(center).T)[:3, 0].tolist()
            )
            heading = heading_transform(box3d, cam2lidar_mat)

            new_box3d = Box3D(
                orientation=(0.0, heading, 0.0),
                location=center_lidar,
                dimension=box3d.dimension,
                alpha=rotation_y_to_alpha(heading, center_lidar),  # type: ignore # pylint: disable=line-too-long
            )
        new_labels.append(
            Label(
                category=label.category,
                box2d=label.box2d,
                box3d=new_box3d if label.box3d is not None else None,
                id=label.id,
            )
        )

    return new_labels


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
) -> Dataset:
    """Function converting kitti detection data to Scalabel format."""
    frames, groups = [], []

    velodyne_dir = osp.join(data_dir, "velodyne")
    left_img_dir = osp.join(data_dir, "image_2")
    label_dir = osp.join(data_dir, "label_2")
    calib_dir = osp.join(data_dir, "calib")

    img_names = sorted(os.listdir(left_img_dir))

    global_track_id = 0

    rotation = (0, 0, 0)
    position = (0, 0, 0)

    cam2global = Extrinsics(location=position, rotation=rotation)

    for img_id, img_name in enumerate(img_names):
        velodyne_name = img_name.split(".")[0] + ".bin"
        trackid_maps: Dict[str, int] = {}
        frame_names = []

        rect = read_calib_det(
            calib_dir, int(img_name.split(".")[0]), 4
        ).reshape(3, 3)
        rect = np.hstack((rect, np.zeros((3, 1))))
        rect = np.vstack((rect, np.array([0.0, 0.0, 0.0, 1.0])))

        velo2cam = read_calib_det(
            calib_dir, int(img_name.split(".")[0]), 5
        ).reshape(3, 4)
        velo2cam = np.vstack((velo2cam, np.array([0.0, 0.0, 0.0, 1.0])))

        for cam_id, cam in enumerate(["image_2", "image_3"]):
            img_dir = osp.join(data_dir, cam)
            with Image.open(osp.join(img_dir, img_name)) as img:
                width, height = img.size
                image_size = ImageSize(height=height, width=width)

            projection = read_calib_det(
                calib_dir, int(img_name.split(".")[0]), cam_id + 2
            ).reshape(3, 4)

            intrinsics = Intrinsics(
                focal=(projection[0][0], projection[1][1]),
                center=(projection[0][2], projection[1][2]),
            )

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

            full_path = osp.join(img_dir, img_name)
            url = data_type + full_path.split(data_type)[-1]

            video_name = "/".join(url.split("/")[:-1])

            f = Frame(
                name=f"{cam}_" + img_name,
                videoName=video_name,
                frameIndex=img_id,
                url=url,
                size=image_size,
                extrinsics=cam2global,
                intrinsics=intrinsics,
                labels=labels,
            )
            frame_names.append(f"{cam}_" + img_name)
            frames.append(f)

        full_path = osp.join(velodyne_dir, velodyne_name)
        url = data_type + full_path.split(data_type)[-1]

        video_name = "/".join(url.split("/")[:-1])

        lidar2cam_mat = np.dot(rect, velo2cam)
        lidar2global_mat = np.dot(
            get_matrix_from_extrinsics(cam2global), lidar2cam_mat
        )
        lidar2cam = get_extrinsics_from_matrix(lidar2cam_mat)
        lidar2global = get_extrinsics_from_matrix(lidar2global_mat)

        groups.append(
            FrameGroup(
                name=velodyne_name.split(".")[0],
                videoName=video_name,
                url=url,
                extrinsics=lidar2global,
                frames=frame_names,
                labels=parse_lidar_labels(labels, lidar2cam),
            )
        )

    cfg = Config(categories=[Category(name=n) for n in kitti_used_cats])
    dataset = Dataset(frames=frames, groups=groups, config=cfg)
    return dataset


def from_kitti(
    data_dir: str,
    data_type: str,
) -> Dataset:
    """Function converting kitti data to Scalabel format."""
    if data_type == "detection":
        return from_kitti_det(data_dir, data_type)

    frames, groups = [], []

    velodyne_dir = osp.join(data_dir, "velodyne")
    label_dir = osp.join(data_dir, "label_02")
    calib_dir = osp.join(data_dir, "calib")
    oxt_dir = osp.join(data_dir, "oxts")

    video_names = sorted(os.listdir(velodyne_dir))

    global_track_id = 0
    total_img_names = {}

    for video_name in video_names:
        for cam in ["image_02", "image_03"]:
            img_dir = osp.join(data_dir, cam)
            total_img_names[cam] = sorted(
                [
                    f.path
                    for f in os.scandir(osp.join(img_dir, video_name))
                    if f.is_file() and f.name.endswith("png")
                ]
            )
        velodyne_names = sorted(
            [
                f.path
                for f in os.scandir(osp.join(velodyne_dir, video_name))
                if f.is_file() and f.name.endswith("bin")
            ]
        )
        trackid_maps: Dict[str, int] = {}

        projections = {}
        projections["image_02"] = read_calib(
            calib_dir, int(video_name), 2
        ).reshape(3, 4)
        projections["image_03"] = read_calib(
            calib_dir, int(video_name), 3
        ).reshape(3, 4)
        rect = read_calib(calib_dir, int(video_name), 4).reshape(3, 3)
        rect = np.hstack((rect, np.zeros((3, 1))))
        rect = np.vstack((rect, np.array([0.0, 0.0, 0.0, 1.0])))

        velo2cam = read_calib(calib_dir, int(video_name), 5).reshape(3, 4)
        velo2cam = np.vstack((velo2cam, np.array([0.0, 0.0, 0.0, 1.0])))

        if osp.exists(label_dir):
            label_file = osp.join(label_dir, f"{video_name}.txt")
            labels_dict, trackid_maps, global_track_id = parse_label(
                data_type, label_file, trackid_maps, global_track_id
            )

        for frame_idx in range(len(total_img_names["image_02"])):
            fields = read_oxts(oxt_dir, int(video_name))
            poses = [KittiPoseParser(fields[i]) for i in range(len(fields))]

            rotation = tuple(
                R.from_matrix(poses[frame_idx].rotation)
                .as_euler("xyz")
                .tolist()
            )
            position = tuple(
                np.array(
                    poses[frame_idx].position - poses[0].position
                ).tolist()
            )
            offset_2_to_3 = (
                projections["image_03"][0, -1] / projections["image_03"][0, 0]
                - projections["image_02"][0, -1]
                / projections["image_02"][0, 0]
            )

            frame_names = []
            for cam, img_names in total_img_names.items():
                img_name = img_names[frame_idx]
                with Image.open(img_name) as img:
                    width, height = img.size
                    image_size = ImageSize(height=height, width=width)

                projection = projections[cam]
                intrinsics = Intrinsics(
                    focal=(projection[0][0], projection[1][1]),
                    center=(projection[0][2], projection[1][2]),
                )
                offset = 0
                if cam == "image_03":
                    offset = offset_2_to_3
                cam2global = Extrinsics(
                    location=(position[0] + offset, position[1], position[2]),
                    rotation=rotation,
                )

                if osp.exists(label_dir):
                    if not frame_idx in labels_dict:
                        labels = []
                    else:
                        labels = labels_dict[frame_idx]
                else:
                    labels = []

                url = data_type + img_name.split(data_type)[-1]

                img_name_list = img_name.split(f"{cam}/")[-1].split("/")

                img_name = f"{cam}_" + "_".join(img_name_list)

                labels_cam = []
                for label in labels:
                    label_cam = copy.deepcopy(label)
                    assert label_cam.box3d is not None
                    label_cam.box3d.location = (
                        label_cam.box3d.location[0] + offset,
                        label_cam.box3d.location[1],
                        label_cam.box3d.location[2],
                    )
                    labels_cam.append(label_cam)

                frame = Frame(
                    name=img_name,
                    videoName=video_name,
                    frameIndex=frame_idx,
                    url=url,
                    size=image_size,
                    extrinsics=cam2global,
                    intrinsics=intrinsics,
                    labels=labels_cam,
                )
                frame_names.append(img_name)
                frames.append(frame)

            url = data_type + velodyne_names[frame_idx].split(data_type)[-1]

            lidar2cam_mat = np.dot(rect, velo2cam)
            lidar2global_mat = np.dot(
                get_matrix_from_extrinsics(cam2global), lidar2cam_mat
            )
            lidar2cam = get_extrinsics_from_matrix(lidar2cam_mat)
            lidar2global = get_extrinsics_from_matrix(lidar2global_mat)

            group_name = "_".join(
                velodyne_names[frame_idx]
                .split(velodyne_dir)[-1]
                .split("/")[1:]
            )
            groups.append(
                FrameGroup(
                    name=group_name,
                    videoName=video_name,
                    frameIndex=frame_idx,
                    url=url,
                    extrinsics=lidar2global,
                    frames=frame_names,
                    labels=parse_lidar_labels(labels, lidar2cam),
                )
            )

    cfg = Config(categories=[Category(name=n) for n in kitti_used_cats])
    dataset = Dataset(frames=frames, groups=groups, config=cfg)
    return dataset


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = f"{args.data_type}_{args.split}.json"

    data_dir = osp.join(args.input_dir, args.data_type, args.split)

    scalabel = from_kitti(
        data_dir,
        data_type=args.data_type,
    )

    save(osp.join(args.output_dir, output_name), scalabel)


if __name__ == "__main__":
    run(parse_arguments())
