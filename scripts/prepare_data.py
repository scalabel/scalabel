import os
from subprocess import Popen, PIPE, STDOUT
import argparse
import glob
from os.path import join
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description='video pre-processing')
    parser.add_argument('--input', '-i', type=str, default='',
                        help='path to the video to be processed')
    parser.add_argument('--tar-dir', '-t', type=str, default='',
                        help='target folder to save the frames')
    parser.add_argument('--fps', '-f', type=int, default=5,
                        help='the target frame rate.')

    # Specify S3 bucket path
    parser.add_argument('--s3', type=str, default='',
                        help='bucket-name/subfolder')

    # Output webroot. Ignore it if you are using s3
    parser.add_argument('--web-root', '-w', type=str, default='',
                        help='webroot used as prefix in the yaml file')

    args = parser.parse_args()
    return args


def prepare_data(args):

    if args.s3:
        s3_setup(args)

    if os.path.isfile(args.input):
        print('Splitting the video ...')

        if len(glob.glob(join(args.tar_dir, '*.jpg'))) > 0:
            print('[ERROR] Target folder is not empty. '
                  'Please specify an empty folder')
            return None

        cmd = "ffmpeg -i {} -r {} -qscale:v 2 {}/{}-%07d.jpg".format(
            args.input, args.fps, args.tar_dir,
            os.path.splitext(os.path.split(args.input)[1])[0]
        )
        print('[RUNNING]', cmd)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
        p.wait()
    else:
        print('Loading all existing images in folder {}'.format(args.input))
        args.tar_dir = args.input

    # create the yaml file
    file_list = sorted(glob.glob(join(args.tar_dir, '*.jpg')))

    yaml_items = [
        {'url': os.path.abspath(img) if not args.web_root else
            join(args.web_root, os.path.basename(img))}
        for img in file_list]

    name = os.path.basename(args.input).split('.')[0]
    output = join(args.tar_dir, name+'.yml')
    with open(output, 'w') as f:
        yaml.dump(yaml_items, f)

    # upload to s3 if needed
    if args.s3:
        upload_files_to_s3(args)

    return output


def upload_files_to_s3(args):
    import boto3
    s3 = boto3.resource('s3')
    bn = args.bucket_name
    file_list = glob.glob(join(args.tar_dir, '*'))
    for f in file_list:
        try:
            s3.Bucket(bn).upload_file(f, join(args.s3_folder,
                                              os.path.basename(f)),
                                      ExtraArgs={'ACL': 'public-read'})
        except Exception as e:
            print('[ERROR] s3 bucket is not properly configured')
            print('[ERROR]', e)
            break


def s3_setup(args):
    import boto3

    args.bucket_name, args.s3_folder = args.s3.split('/')[0], '/'.join(
        args.s3.split('/')[1:])
    s3 = boto3.resource('s3')
    bn = args.bucket_name
    region = s3.meta.client.get_bucket_location(
        Bucket=bn)['LocationConstraint']

    args.web_root = join('https://s3-{}.amazonaws.com'.format(region),
                         bn, args.s3_folder)


def main():
    args = parse_arguments()
    if args.tar_dir:
        os.makedirs(args.tar_dir, exist_ok=True)
    output = prepare_data(args)
    if output is not None:
        print('[SUCCESS] The configuration file saved at {}'.format(output))
    else:
        print('[FAIL] Rerun the code.')


if __name__ == '__main__':
    main()
