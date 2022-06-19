"""launches node and redis on a single instance with one node process."""
import argparse
import logging
import os
import subprocess

import psutil
import os
import yaml

import scalabel.automatic.consts.redis_consts as RedisConsts


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)


def get_redis_conf():
    return os.path.join("app", "config", "redis.conf")


def launch() -> None:
    """Launch scalabel processes."""
    logger.info("Launching redis server")
    redis_command = ["redis-server", get_redis_conf(),
                     "--port", str(RedisConsts.REDIS_PORT),
                     "--bind", RedisConsts.REDIS_HOST,
                     "--dir", "./",
                     "--protected-mode", "yes"]
    subprocess.Popen(redis_command)

    logger.info("Launching model server")
    py_command = ["python3.8", "-m", "scalabel.automatic.servers.server"]
    subprocess.call(py_command)


if __name__ == "__main__":
    launch()
