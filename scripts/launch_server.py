""" launches node and redis """
import subprocess
import argparse

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Launch the server.')
    PARSER.add_argument(
        '--config', dest='config',
        help='path to config file', default='./data/config.yml')
    ARGS = PARSER.parse_args()

    subprocess.Popen(["redis-server"])
    subprocess.call(['node', 'app/dist/js/main.js', '--config', ARGS.config])
    