"""Prepare the data folder and list.

Convert videos or images to a data folder and an image list that can be
directly used for creating scalabel projects. Assume all the images are in
`.jpg` format.
"""
from __future__ import annotations
import argparse
import glob
import os
from os.path import join
import shutil
from subprocess import Popen, PIPE, STDOUT
from typing import Union
from dataclasses import dataclass

import boto3
from tqdm import tqdm
import yaml

from scalabel.common.logger import logger


def parse_arguments() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        help="path to the video/images to be processed",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="",
        help="output folder to save the frames",
    )
    parser.add_argument(
        "--fps", "-f", type=int, default=5, help="the target frame rate."
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
                "Invalid `input` value `%s`. Neither file nor directory ", p,
            )
            return None
    return None


def process_video(
    filepath: str, fps: int, start_time: int, max_frames: int, out_dir: str
) -> None:
    """Break one video into a folder of images."""
    if not check_video_format(filepath):
        logger.warning("Ignore invalid file %s", filepath)
        return

    if not os.path.exists(filepath):
        logger.warning("%s does not exist", filepath)
        return
    cmd_args = ["-i {} -r {} -qscale:v 2".format(filepath, fps)]
    if start_time > 0:
        cmd_args.append("-ss {}".format(start_time))
    if max_frames > 0:
        cmd_args.append("-frames:v {}".format(max_frames))
    video_name = os.path.splitext(os.path.split(filepath)[1])[0]
    out_dir = join(out_dir, video_name)
    os.makedirs(out_dir)
    cmd = "ffmpeg {0} {1}/{2}-%07d.jpg".format(
        " ".join(cmd_args), out_dir, video_name
    )
    logger.info("RUNNING %s", cmd)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
    p.wait()


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


def prepare_data(args: argparse.Namespace) -> Union[str, None]:
    """Break one or a list of videos into frames."""
    url_root = args.url_root
    if args.s3:
        url_root = s3_setup(args.s3)

    logger.info("processing %d video(s) ...", len(args.input))

    for i in range(len(args.input)):
        filepath = args.input[i]

        if os.path.isdir(filepath):
            copy_images(filepath, args.out_dir)

        elif os.path.isfile(filepath):
            process_video(
                filepath,
                args.fps,
                args.start_time,
                args.max_frames,
                args.out_dir,
            )

    # create the yaml file
    file_list = sorted(glob.glob(join(args.out_dir, "*/*.jpg")))

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

    output = join(args.out_dir, "image_list.yml")
    with open(output, "w") as f:
        yaml.dump(yaml_items, f)

    # upload to s3 if needed
    if args.s3:
        upload_files_to_s3(args.s3, args.out_dir)

    return output


@dataclass
class S3Param:
    """S3 parameters."""

    bucket: str
    folder: str


def parse_s3_path(s3_path: str) -> S3Param:
    """Parse s3 input path into s3 param."""
    return S3Param(
        bucket=s3_path.split("/")[0], folder="/".join(s3_path.split("/")[1:]),
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
        os.makedirs(args.out_dir)

    output = prepare_data(args)
    if output is not None:
        logger.info("The configuration file saved at %s", output)
    else:
        logger.info("Rerun the code.")


if __name__ == "__main__":
    main()
