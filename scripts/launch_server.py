""" launches node and redis """
import subprocess
import argparse
import yaml


def main() -> None:
    """ main process launcher """
    parser = argparse.ArgumentParser(description='Launch the server.')
    parser.add_argument(
        '--config', dest='config',
        help='path to config file', default='./data/config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    if 'redisPort' not in config:
        config['redisPort'] = 6379
    subprocess.Popen(
        ['redis-server', '--port', '{}'.format(config['redisPort'])])
    subprocess.call(['node', 'app/dist/js/main.js', '--config', args.config])


if __name__ == '__main__':
    main()
