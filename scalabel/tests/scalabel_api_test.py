""" Test file for the scalabel api """
import subprocess
import time
import logging
import os
import signal
import shutil
from typing import Iterator, Mapping, Union
import pytest
import yaml
from urllib3.exceptions import HTTPError
from scalabel.scalabel_api import ScalabelAPI

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

CONFIG_PATH = './app/config/test_config.yml'
SERVER_FIXTURE_NAME = 'run_server'

ConfigType = Mapping[str, Union[str, int]]


@pytest.fixture(name='config')  # type: ignore
def fixture_get_config() -> ConfigType:
    """ Read testing config """
    with open(CONFIG_PATH, 'r') as fp:
        config: ConfigType = yaml.load(fp, Loader=yaml.FullLoader)
        return config


@pytest.fixture(name='api')  # type: ignore
def fixture_get_api(config: ConfigType) -> ScalabelAPI:
    """ Get instance of scalabel api """
    port = config['port']
    assert isinstance(port, int)
    return ScalabelAPI(port)


@pytest.yield_fixture(name=SERVER_FIXTURE_NAME)  # type: ignore
def fixture_run_server(
        config: ConfigType) -> Iterator[subprocess.Popen]:  # type: ignore
    """ Setup and teardown node server """
    # create the test dir
    data = config['data']
    assert isinstance(data, str)
    os.makedirs(data)

    server_cmd = ['python3.8', 'scripts/launch_server.py',
                  '--config', CONFIG_PATH]

    logger.info('Launching testing server')
    logger.info(' '.join(server_cmd))
    process = subprocess.Popen(  # pylint: disable=subprocess-popen-preexec-fn
        server_cmd, preexec_fn=os.setsid)
    time.sleep(1)
    yield process

    # kill all subprocesses
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.kill()

    # clean up test data dir
    shutil.rmtree(data)


@pytest.mark.usefixtures(SERVER_FIXTURE_NAME)  # type: ignore
def test_create_project(api: ScalabelAPI) -> None:
    """ Test project creation internal API """
    project_name = 'internal_project'
    api.create_default_project(project_name)

    # test repeated name fails
    with pytest.raises(HTTPError):
        api.create_default_project(project_name)


@pytest.mark.usefixtures(SERVER_FIXTURE_NAME)  # type: ignore
def test_create_project_no_items(api: ScalabelAPI) -> None:
    """ Test internal project creation API allows adding items later """
    project = api.create_default_project('other_project', False)

    # now add the items
    project.add_default_tasks()
