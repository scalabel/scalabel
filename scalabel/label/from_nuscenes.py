"""Conversion script for nuscenes to scalabel."""
import os
import argparse
from .typing import Frame, Box2D, Box3D, Intrinsics, Extrinsics, ImageSize
try:
    import nuscenes as nu
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.geometry_utils import box_in_image, view_points, transform_matrix
    from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
    from nuscenes.utils.data_classes import Quaternion, LidarPointCloud
except ImportError:
    nu = None

cams = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


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
        help="NuScenes dataset version to convert.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for Scalabel format annotations.",
    )
    parser.add_argument(
        "--save-images",
        "-s",
        action="store_true",
        help="If the images should be extracted from .tfrecords and saved."
        "(necessary for using Waymo Open data with Scalabel format "
        "annotations)",
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
    records = [(data.get('sample', record['first_sample_token'])['timestamp'], record) for record in data.scene]
    entries = []

    for start_time, record in sorted(records):
        start_time = data.get('sample', record['first_sample_token'])['timestamp'] / 1000000
        token = record['token']
        name = record['name']
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record['name'].split("-")[:2])
        first_sample_token = record["first_sample_token"]

        entries.append((host, name, date, token, first_sample_token))

    dataframe = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
    return data, dataframe


def quaternion_to_yaw(q: Quaternion, in_image_frame: bool = True) -> float:
    """Convert quaternion angle representation to yaw."""
    if in_image_frame:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
    else:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

    return yaw


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle  to quaternion representation."""
    return Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).elements


def move_boxes_to_camera_space(boxes, ego_pose, cam_pose) -> None:
    """Move boxes from world space to car space to cam space.

    Note: mutates input boxes.
    """
    translation_car = -np.array(ego_pose['translation'])
    rotation_car = Quaternion(ego_pose['rotation']).inverse

    translation_cam = -np.array(cam_pose['translation'])
    rotation_cam = Quaternion(cam_pose['rotation']).inverse

    for box in boxes:
        # Bring box to car space
        box.translate(translation_car)
        box.rotate(rotation_car)

        # Bring box to cam space
        box.translate(translation_cam)
        box.rotate(rotation_cam)


def parse_labels(boxes, calibration_cam, ego_pose_ref, img_size: Tuple[int, int]) -> Optional[List[Labels]]:
    """Parse NuScenes LiDAR labels into single camera."""
    if len(boxes):
        labels = []
        # transform into the camera coord system
        move_boxes_to_camera_space(boxes, ego_pose_ref, calibration_cam)
        intrinsics = np.asarray(calibration_cam["camera_intrinsic"])
        for i in range(len(boxes)):
            box_class = category_to_detection_name(boxes[i].name)
            if box_in_image(boxes[i], intrinsics, img_size) and box_class is not None:
                xyz = boxes[i].center
                w, l, h = boxes[i].wlh
                roty = quaternion_to_yaw(boxes[i].orientation)
                # Project 3d box to 2d.
                corners = boxes[i].corners()
                corner_coords = view_points(corners, intrinsics, True).T[:, :2].tolist()
                # Keep only corners that fall within the image.
                x1, y1, x2, y2 = post_process_coords(corner_coords)
                instance_token = data.get('sample_annotation', boxes[i].token)['instance_token']
                label = Label(
                    id=instance_token,
                    category=box_class,
                    box2d=Box2D(x1=x1, y1=y1, x2=x2, y2=y2),
                    box3d=Box3D(location=xyz, dimension=(h, w, l), orientation=(0, roty, 0)),
                )
                labels.append(label)

        return labels
    else:
        return None

def parse_sequence(first_sample_token: str, scene_name: str) -> List[Frame]:
    """Parse a full nuscenes sequence and convert it into scalabel frames."""
    sample_token = first_sample_token
    while sample_token:
        for cam in cams:
            sample = data.get("sample", sample_token)
            lidar_token = (sample["data"]["LIDAR_TOP"], sample)
            cam_token = sample["data"][cam]
            cam_data = data.get("sample_data", cam_token)
            ego_pose_ref = data.get("ego_pose", cam_data["ego_pose_token"])
            cam_filepath = data.get_sample_data_path(cam_token)
            img_wh = (cam_data['width'], cam_data['height'])
            calibration_cam = data.get("calibrated_sensor",
                                       cam_data["calibrated_sensor_token"])

            boxes = data.get_boxes(lidar_token[0])
            labels = parse_labels(boxes, calibration_cam, ego_pose_ref, img_wh)
            # TODO add other attributes
            # however here is not clear how to deal with multi-cam case and
            # LiDAR pointclouds

            frame = Frame(size=ImageSize(width=img_wh[0], height=img_wh[1]), labels=labels)

        sample_token = sample["next"]

def from_nuscenes(
        data_path: str,
        version: str,
        output_dir: str,
        save_images: bool = False,
        nproc: int = NPROC,
) -> List[Frame]:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    data, df = load_data(data_path, version)
    func = partial(parse_record, output_dir, save_images, use_lidar_labels)
    partial_frames = pmap(
        func,
        zip(df.first_sample_token.values, df.scene_name.values),
        nprocs=nproc,
    )
    frames = []
    for f in partial_frames:
        frames.extend(f)
    return frames


def run(args: argparse.Namespace) -> None:
    """Run conversion with command line arguments."""
    assert nu is not None, (
        "Please install the requirements in scripts/optional.txt to use"
        "NuScenes conversion."
    )
    result = from_nuscenes(
        args.input,
        args.version,
        args.output,
        args.save_images,
        args.nproc,
    )
    save(os.path.join(args.output, "scalabel_anns.json"), result)


if __name__ == "__main__":
    run(parse_arguments())
