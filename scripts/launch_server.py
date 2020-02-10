""" launches node and redis """
import subprocess
import argparse
<<<<<<< HEAD

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Launch the server.')
    PARSER.add_argument(
        '--config', dest='config',
        help='path to config file', default='./data/config.yml')
    ARGS = PARSER.parse_args()

    subprocess.Popen(["redis-server"])
    subprocess.call(['node', 'app/dist/js/main.js', '--config', ARGS.config])
    
=======
import yaml


def main() -> None:
    """ main process launcher """
    parser = argparse.ArgumentParser(
        description='Launch the server on one machine.')
    parser.add_argument(
        '--config', dest='config',
        help='path to config file', default='./data/config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    if 'redisPort' not in config:
        config['redisPort'] = 6379
    subprocess.Popen(
        ['redis-server', '--port', '{}'.format(config['redisPort']),
         '--bind', '127.0.0.1', '--protected-mode', 'yes'])
    subprocess.call(['node', 'app/dist/js/main.js', '--config', args.config])


if __name__ == '__main__':
    main()
>>>>>>> master
