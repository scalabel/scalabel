"""
Convert videos or images to a data folder and an image list that can be
directly used for creating scalabel projects
"""
from __future__ import annotations
import argparse
import glob
import os
from os.path import join
import shutil
from subprocess import Popen, PIPE, STDOUT
from typing import Union

import boto3
import yaml


def parse_arguments() -> argparse.Namespace:
    "parse arguments"
    parser = argparse.ArgumentParser(description='prepare data')
    parser.add_argument('--input', '-i', type=str, nargs='+',
                        help='path to the video/images to be processed')
    parser.add_argument('--tar-dir', '-t', type=str, default='',
                        help='target folder to save the frames')
    parser.add_argument('--fps', '-f', type=int, default=5,
                        help='the target frame rate.')
    parser.add_argument('--scratch', action='store_true',
                        help='ignore non-empty folder.')

    # Specify S3 bucket path
    parser.add_argument('--s3', type=str, default='',
                        help='bucket-name/subfolder')

    # Output webroot. Ignore it if you are using s3
    parser.add_argument('--web-root', '-w', type=str, default='',
                        help='webroot used as prefix in the yaml file')

    args = parser.parse_args()
    return args


def check_video_format(name: str) -> bool:
    "only accept mov, avi, and mp4"
    if name.endswith(('.mov', '.avi', '.mp4')):
        return True
    return False


def prepare_data(args: argparse.Namespace) -> Union[str, None]:
    "break one or a list of videos into frames"
    if args.s3:
        s3_setup(args)

    if isinstance(args.input, list):
        print('processing {} video(s) ...'.format(len(args.input)))
        if glob.glob(join(args.tar_dir, '*.jpg')):
            print('[ERROR] Target folder is not empty. '
                  'Please specify an empty folder')
            return None

        for i in range(len(args.input)):
            vf = args.input[i]

            if os.path.isdir(vf):
                print('Loading all existing images in folder {}'.format(vf))
                args.tar_dir = vf

            elif os.path.isfile(vf):
                if not check_video_format(vf):
                    print('[WARNING] Ignore invalid file {}'.format(vf))
                    continue

                if not os.path.exists(vf):
                    print('[WARNING] {} does not exist'.format(vf))
                    continue

                cmd = "ffmpeg -i {} -r {} -qscale:v 2 {}/{}-%07d.jpg".format(
                    vf, args.fps, args.tar_dir,
                    os.path.splitext(os.path.split(vf)[1])[0]
                )
                print('[RUNNING]', cmd)
                p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
                p.wait()
            else:
                print('[ERROR] Invalid `input` value `{}`. Neither file '
                      'nor directory '.format(vf))
                return None

    # create the yaml file
    file_list = sorted(glob.glob(join(args.tar_dir, '*.jpg')))

    yaml_items = [
        {
            'url': (os.path.abspath(img) if not args.web_root else
                    join(args.web_root, os.path.basename(img))),
            'videoName': '{}'.format(
                '-'.join(os.path.split(img)[-1].split('-')[:-1]))
        }
        for img in file_list]

    output = join(args.tar_dir, 'image_list.yml')
    with open(output, 'w') as f:
        yaml.dump(yaml_items, f)

    # upload to s3 if needed
    if args.s3:
        upload_files_to_s3(args)

    return output


def upload_files_to_s3(args: argparse.Namespace) -> None:
    """send the files to s3"""
    s3 = boto3.resource('s3')
    bn = args.bucket_name
    file_list = glob.glob(join(args.tar_dir, '*'))
    for f in file_list:
        try:
            # pylint is added here because it thinks boto3.resource is a string
            s3.Bucket(bn).upload_file(f, join(args.s3_folder,  # pylint: disable=no-member
                                              os.path.basename(f)),
                                      ExtraArgs={'ACL': 'public-read'})
        except boto3.exceptions.S3UploadFailedError as e:
            print('[ERROR] s3 bucket is not properly configured')
            print('[ERROR]', e)
            break


def s3_setup(args: argparse.Namespace) -> None:
    """optionaly store the data on s3"""
    args.bucket_name, args.s3_folder = args.s3.split('/')[0], '/'.join(
        args.s3.split('/')[1:])
    s3 = boto3.resource('s3')
    bn = args.bucket_name
    region = s3.meta.client.get_bucket_location(
        Bucket=bn)['LocationConstraint']

    args.web_root = join('https://s3-{}.amazonaws.com'.format(region),
                         bn, args.s3_folder)


def main() -> None:
    """ main function """
    args = parse_arguments()
    if args.tar_dir:
        if args.scratch and os.path.exists(args.tar_dir):
            print('[INFO] Remove existing target directory')
            shutil.rmtree(args.tar_dir)
        os.makedirs(args.tar_dir)

    output = prepare_data(args)
    if output is not None:
        print('[SUCCESS] The configuration file saved at {}'.format(output))
    else:
        print('[FAIL] Rerun the code.')


if __name__ == '__main__':
    main()
