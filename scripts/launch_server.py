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

    # store redis dump in current directory if no local dir is supplied
    data = 'data'
    database = 'database'
    if (data not in config or
            (database in config and config[database] != 'local')):
        config[data] = './'

    redis_cmd = ['redis-server', 'app/config/redis.conf',
                 '--port', '{}'.format(config[redis_port]),
                 '--bind', '127.0.0.1',
                 '--dir', config[data],
                 '--protected-mode', 'yes']
    logger.info('Launching redis server')
    logger.info(' '.join(redis_cmd))
    subprocess.Popen(redis_cmd)

    # launch the python server if bot option is true
    bot = 'bots'
    if bot in config and config[bot]:
        py_command = ['python3.6', 'scalabel/server.py']
        host = 'pyHost'
        port = 'pyPort'
        if host in config:
            py_command += ['--host', config[host]]
        if port in config:
            py_command += ['--port', str(config[port])]

        logger.info('Launching python server')
        logger.info(' '.join(py_command))
        subprocess.Popen(py_command)

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
