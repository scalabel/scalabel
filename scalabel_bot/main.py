# -*- coding: utf-8 -*-
"""PipeSwitch Run Script

This module is the main entry point for the PipeSwitch Manager.

Run this file from the main directory of the project::

    $ python3 main.py [-h] [--args]

For profiling, run:

    $ py-spy top --pid <PID> --subprocesses --nonblocking
    $ python profiling/profile.py [-h] [--args]

Todo:
    * None
"""

import os
import shutil
import traceback
from argparse import ArgumentParser
from torch.multiprocessing import set_start_method
from redis import exceptions, Redis
import ast

from scalabel_bot.common.consts import (
    DEBUG_LOG_FILE,
    REDIS_HOST,
    REDIS_PORT,
    TIMING_LOG_FILE,
)

from scalabel_bot.common.logger import logger
from scalabel_bot.manager.bot_manager import BotManager


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="PipeSwitch Run Script")

    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help=(
            "Mode of the BotManager. Setting the flag uses CPU execution"
            " instead of GPU. Default is False"
        ),
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use. Default is 1.",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="",
        help=("Specific GPU ids"),
    )
    parser.add_argument(
        "--redis",
        action="store_true",
        default=False,
        help="Whether to start a local Redis server. Default is False.",
    )
    return parser


def clear_logs(file: str) -> None:
    if not os.path.exists(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    if os.stat(file).st_size != 0:
        archive_file = os.path.join(
            os.path.dirname(file),
            f"{os.stat(file).st_mtime}.log",
        )
        _ = shutil.copyfile(file, archive_file)
    with open(file, "w", encoding="utf-8") as f:
        f.write("")


def launch():
    try:
        clear_logs(DEBUG_LOG_FILE)
        clear_logs(TIMING_LOG_FILE)
        logger.info(f"PID: {os.getpid()}")
        args: ArgumentParser = get_parser().parse_args()
        logger.info(f"Arguments: {str(args)}")
        if args.cpu:
            mode = "cpu"
        else:
            mode = "gpu"
        if args.redis:
            os.system("redis-server scalabel_bot/redis.conf")
        redis = Redis(host=REDIS_HOST, port=REDIS_PORT)
        if not redis.ping():
            logger.warning("Cannot connect to Redis server.")
            logger.warning("Please restart Redis.")
            return
        if args.gpu_id:
            gpu_ids = ast.literal_eval(args.gpu_id)
        else:
            gpu_ids = []
        manager: BotManager = BotManager(
            mode=mode, num_gpus=args.num_gpus, gpu_ids=gpu_ids
        )
        manager.run()
    except exceptions.ConnectionError as conn_err:
        logger.error(conn_err)
        logger.warning("Redis server is not running")
        logger.warning(
            "Please start it before running the script, or add the '--redis'"
            " flag to automatically start a local Redis server."
        )
    except BrokenPipeError as err:
        logger.error(err)
    except ConnectionResetError as err:
        logger.error(err)
    except KeyboardInterrupt:
        pass
    except Exception as err:  # pylint: disable=broad-except
        logger.error(err)
        logger.error(traceback.format_exc())
    finally:
        if "manager" in locals():
            manager.shutdown()
        if args.redis:
            os.system("redis-cli shutdown")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    launch()
