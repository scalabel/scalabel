""" Test file for the scalabel api """
import subprocess
import time
import logging
import os
import signal
import shutil
import pytest
import yaml
from ..scalabel_api import ScalabelAPI

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

CONFIG_PATH = './app/config/test_config.yml'
SERVER_FIXTURE_NAME = 'run_server'


@pytest.fixture(name='config')
def fixture_get_config():
    """ Read testing config """
    with open(CONFIG_PATH, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


@pytest.fixture(name='api')
def fixture_get_api(config):
    """ Get instance of scalabel api """
    return ScalabelAPI(config['port'])


@pytest.yield_fixture(name=SERVER_FIXTURE_NAME)
def fixture_run_server(config):
    """ Setup and teardown node server """
    server_cmd = ['python3.8', 'scripts/launch_server.py',
                  '--config', CONFIG_PATH]

    logger.info('Launching testing server')
    logger.info(' '.join(server_cmd))
    process = subprocess.Popen(server_cmd, preexec_fn=os.setsid)
    time.sleep(1)
    yield process

    # kill all subprocesses
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.kill()

    # clean up data dir
    shutil.rmtree(config['data'])


@pytest.mark.usefixtures(SERVER_FIXTURE_NAME)
def test_create_project(api):
    """ Test project creation internal API """
    project_name = 'internal_project'
    response = api.create_default_project(project_name)
    assert response.status_code == 200

    # test repeated name fails
    response = api.create_default_project(project_name)
    assert response.status_code == 400


@pytest.mark.usefixtures(SERVER_FIXTURE_NAME)
def test_create_project_no_items(api):
    """ Test internal project creation API allows adding items later """
    project_name = 'other_project'
    response = api.create_default_project(project_name, False)
    assert response.status_code == 200

    # now add the items
    response = api.add_default_tasks(project_name)
    assert response.status_code == 200
