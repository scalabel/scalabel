""" launches node and redis """
import subprocess
import argparse

def main():
    """ main process launcher """
    parser = argparse.ArgumentParser(description='Launch the server.')
    parser.add_argument(
        '--config', dest='config',
        help='path to config file', default='./data/config.yml')
    args = parser.parse_args()

    subprocess.Popen(["redis-server"])
    subprocess.call(['node', 'app/dist/js/main.js', '--config', args.config])

if __name__ == '__main__':
    main()
    