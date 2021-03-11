"""launches node and redis on a single instance with one node process."""
import argparse
import logging
import os
import subprocess

import psutil
import yaml

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)


def launch() -> None:
    """Launch scalabel processes."""
    logger.info("Launching Scalabel server")
    parser = argparse.ArgumentParser(
        description="Launch the server on one machine."
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="path to config file",
        default="./data/config.yml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # launch the python server if bot option is true
    bot = "bots"
    if bot in config and config[bot]:
        py_command = ["python3.8", "-m", "scalabel.bot.server"]

        host = "botHost"
        port = "botPort"
        if host in config:
            py_command += ["--host", config[host]]
        if port in config:
            py_command += ["--port", str(config[port])]

        py_env = os.environ.copy()
        python_path = "PYTHONPATH"
        model_path = os.path.join(
            "scalabel",
            "bot",
            "experimental",
            "fast-seg-label",
            "polyrnn_scalabel",
        )
        if python_path in py_env:
            model_path = "{}:{}".format(py_env[python_path], model_path)
        py_env[python_path] = model_path

        logger.info("Launching python server")
        logger.info(" ".join(py_command))

        subprocess.Popen(py_command, env=py_env)

    # Try to use all the available memory for this single instance launcher
    memory = psutil.virtual_memory()
    max_memory = int(memory.available / 1024 / 1024)
    node_cmd = [
        "node",
        "app/dist/main.js",
        "--config",
        args.config,
        "--max-old-space-size={}".format(max_memory),
    ]
    logger.info("Launching nodejs")
    logger.info(" ".join(node_cmd))
    subprocess.call(node_cmd)


if __name__ == "__main__":
    launch()
