""" launches node and redis on a single instance with one node process """
import subprocess
import argparse
import logging
import psutil
import yaml


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)


def launch() -> None:
    """ main process launcher """
    logger.info('Launching Scalabel server')
    parser = argparse.ArgumentParser(
        description='Launch the server on one machine.')
    parser.add_argument(
        '--config', dest='config',
        help='path to config file', default='./data/config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    redis_port = 'redisPort'
    if redis_port not in config:
        config[redis_port] = 6379
    redis_cmd = ['redis-server', '--port', '{}'.format(config[redis_port]),
                 '--bind', '127.0.0.1', '--protected-mode', 'yes']
    logger.info('Launching redis server')
    logger.info(' '.join(redis_cmd))
    subprocess.Popen(redis_cmd)

    # Try to use all the available memory for this single instance launcher
    memory = psutil.virtual_memory()
    max_memory = int(memory.available / 1024 / 1024)
    node_cmd = ['node', 'app/dist/js/main.js', '--config', args.config,
                '--max-old-space-size={}'.format(max_memory)]
    logger.info('Launching nodejs')
    logger.info(' '.join(node_cmd))
    subprocess.call(node_cmd)


if __name__ == '__main__':
    launch()
