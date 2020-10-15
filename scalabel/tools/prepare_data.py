"""Prepare the data folder and list.

Convert videos or images to a data folder and an image list that can be
directly used for creating scalabel projects. Assume all the images are in
`.jpg` format.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from dataclasses import dataclass
from os.path import join
from subprocess import DEVNULL, check_call
from typing import List

import boto3
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from scalabel.common.logger import logger


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=[],
        nargs="+",
        required=True,
        help="path to the video/images to be processed",
    )
    parser.add_argument(
        "--input-list",
        type=str,
        nargs="+",
        default=[],
        help="List of input directories and videos for processing."
        " Each line in each file is a file path.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="",
        required=True,
        help="output folder to save the frames",
    )
    parser.add_argument(
        "--fps",
        "-f",
        type=float,
        default=0,
        help="the target frame rate. default is the original video framerate.",
    )
    parser.add_argument(
        "--scratch", action="store_true", help="ignore non-empty folder."
    )

    # Specify S3 bucket path
    parser.add_argument(
        "--s3",
        type=str,
        default="",
        help="Specify S3 bucket path in bucket-name/subfolder",
    )

    # Output webroot. Ignore it if you are using s3
    parser.add_argument(
        "--url-root",
        type=str,
        default="",
        help="Url root used as prefix in the yaml file. "
        "Ignore it if you are using s3.",
    )

    parser.add_argument(
        "--start-time",
        type=int,
        default=0,
        help="The starting time of extracting frames in seconds.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Max number of output frames for each video.",
    )

    parser.add_argument(
        "--no-list",
        action="store_true",
        help="do not generate the image list.",
    )

    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=0,
        help="Process multiple videos in parallel.",
    )

    args = parser.parse_args()
    return args


def check_video_format(name: str) -> bool:
    """Accept only mov, avi, and mp4."""
    if name.endswith((".mov", ".avi", ".mp4")):
        return True
    return False


def check_args(args: argparse.Namespace) -> None:
    """Validate input arguments."""
    if glob.glob(join(args.out_dir, "*.jpg")):
        logger.error(
            "Target folder is not empty. Please specify an empty folder"
        )
        return None
    for p in args.input:
        if not (os.path.isdir(p) or os.path.isfile(p)):
            logger.error(
                "Invalid `input` value `%s`. Neither file nor directory ",
                p,
            )
            return None
    return None


def process_video(
    filepath: str,
    fps: float,
    start_time: int,
    max_frames: int,
    out_dir: str,
    quiet: bool = False,
) -> None:
    """Break one video into a folder of images."""
    if not check_video_format(filepath):
        logger.warning("Ignore invalid file %s", filepath)
        return

    if not os.path.exists(filepath):
        logger.warning("%s does not exist", filepath)
        return
    cmd = ["ffmpeg", "-i", filepath, "-qscale:v", "2"]
    if start_time > 0:
        cmd.extend(["-ss", str(start_time)])
    if max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    if fps > 0:
        cmd.extend(["-r", str(fps)])
    video_name = os.path.splitext(os.path.split(filepath)[1])[0]
    out_dir = join(out_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)
    cmd.append("{}/{}-%07d.jpg".format(out_dir, video_name))
    if not quiet:
        logger.info("RUNNING %s", cmd)
    pipe = DEVNULL if quiet else None
    check_call(cmd, stdout=pipe, stderr=pipe)


def create_image_list(out_dir: str, url_root: str) -> str:
    """Create image list from the output directory."""
    file_list = sorted(glob.glob(join(out_dir, "*/*.jpg")))

    yaml_items = [
        {
            "url": (
                os.path.abspath(img)
                if not url_root
                else join(url_root, img.split("/")[-2], os.path.basename(img))
            ),
            "videoName": img.split("/")[-2],
        }
        for img in file_list
    ]

    list_path = join(out_dir, "image_list.yml")
    with open(list_path, "w") as f:
        yaml.dump(yaml_items, f)
    logger.info(
        "The configuration file saved at %s with %d items",
        list_path,
        len(file_list),
    )
    return list_path


def copy_images(in_path: str, out_path: str) -> None:
    """Copy images to the output folder with proper video name."""
    file_list = glob.glob(join(in_path, "**/*.jpg"), recursive=True)
    in_path_parts = os.path.split(in_path)
    if len(in_path_parts) > 0 and len(in_path_parts[-1]) > 0:
        video_name = in_path_parts[-1]
    else:
        video_name = "default"
    out_dir = join(out_path, video_name)
    os.makedirs(out_dir)
    logger.info("Copying %s to %s", in_path, out_dir)
    for image_path in tqdm(file_list):
        image_name = os.path.split(image_path)[-1]
        shutil.copyfile(image_path, join(out_dir, image_name))


def process_input(
    filepath: str,
    fps: float,
    start_time: int,
    max_frames: int,
    out_dir: str,
    quiet: bool = False,
) -> None:
    """Process one input folder or video."""
    if os.path.isdir(filepath):
        copy_images(filepath, out_dir)

    elif os.path.isfile(filepath):
        process_video(
            filepath,
            fps,
            start_time,
            max_frames,
            out_dir,
            quiet,
        )


def parse_input_list(args: argparse.Namespace) -> List[str]:
    """Get all the input paths from args."""
    inputs = []
    inputs.extend(args.input)
    for l in args.input_list:
        with open(l, "r") as fp:
            inputs.extend([l.strip() for l in fp.readlines()])
    return inputs


def prepare_data(args: argparse.Namespace) -> None:
    """Break one or a list of videos into frames."""
    url_root = args.url_root
    if args.s3:
        url_root = s3_setup(args.s3)

    inputs = parse_input_list(args)
    logger.info("processing %d video(s) ...", len(inputs))

    num_videos = len(inputs)
    video_range = range(len(inputs))
    quiet = num_videos > 1
    if num_videos > 1:
        video_range = tqdm(video_range)
    jobs = args.jobs
    if num_videos >= jobs > 0:
        Parallel(n_jobs=jobs, backend="multiprocessing")(
            delayed(process_input)(
                inputs[i],
                args.fps,
                args.start_time,
                args.max_frames,
                args.out_dir,
                quiet,
            )
            for i in video_range
        )
    else:
        for i in video_range:
            process_input(
                inputs[i],
                args.fps,
                args.start_time,
                args.max_frames,
                args.out_dir,
                quiet,
            )

    # upload to s3 if needed
    if args.s3:
        upload_files_to_s3(args.s3, args.out_dir)

    # create the yaml file
    if not args.no_list:
        create_image_list(args.out_dir, url_root)


@dataclass
class S3Param:
    """S3 parameters."""

    bucket: str
    folder: str


def parse_s3_path(s3_path: str) -> S3Param:
    """Parse s3 input path into s3 param."""
    return S3Param(
        bucket=s3_path.split("/")[0],
        folder="/".join(s3_path.split("/")[1:]),
    )


def upload_files_to_s3(s3_path: str, out_dir: str) -> None:
    """Send the files to s3."""
    s3 = boto3.resource("s3")
    s3_param = parse_s3_path(s3_path)
    file_list = glob.glob(join(out_dir, "**/*.jpg"), recursive=True)
    logger.info(
        "Uploading %d files to s3 %s:%s",
        len(file_list),
        s3_param.bucket,
        s3_param.folder,
    )
    for f in tqdm(file_list):
        try:
            # pylint is added here because it thinks boto3.resource is a string
            s3.Bucket(s3_param.bucket).upload_file(
                f,
                join(s3_param.folder, f[len(out_dir) + 1 :]),
                ExtraArgs={"ACL": "public-read"},
            )
        except boto3.exceptions.S3UploadFailedError as e:
            logger.error("s3 bucket is not properly configured %s", e)
            break


def s3_setup(s3_path: str) -> str:
    """Store optionaly the data on s3."""
    s3_param = parse_s3_path(s3_path)
    s3 = boto3.resource("s3")
    region = s3.meta.client.get_bucket_location(Bucket=s3_param.bucket)[
        "LocationConstraint"
    ]

    return join(
        "https://s3-{}.amazonaws.com".format(region),
        s3_param.bucket,
        s3_param.folder,
    )


def main() -> None:
    """Run main function."""
    args = parse_arguments()
    if args.out_dir:
        if args.scratch and os.path.exists(args.out_dir):
            logger.info("Remove existing target directory")
            shutil.rmtree(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)

    prepare_data(args)


if __name__ == "__main__":
    main()
