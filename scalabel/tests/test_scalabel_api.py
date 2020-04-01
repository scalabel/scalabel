import subprocess
import requests
import yaml
import pytest
import time
import logging
import os
import signal
import json
import shutil

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

CONFIG_PATH = './app/config/test_config.yml'
HEADERS = {'Content-Type': 'application/json'}
SERVER_FIXTURE_NAME = 'run_server'


@pytest.fixture(name='get_config')
def fixture_get_config():
    """ read testing config """
    with open(CONFIG_PATH, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


@pytest.yield_fixture(name=SERVER_FIXTURE_NAME)
def fixture_run_server(get_config):
    """ setup and teardown node server """
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
    shutil.rmtree(get_config['data'])


def get_sample_fields(project_name):
    """ creates dict with required fields """
    return {
        'project_name': project_name,
        'item_type': 'image',
        'label_type': 'box2d',
        'task_size': 10
    }


def get_create_uri(port):
    """ gets endpoint for project creation given the port """
    address = '127.0.0.1'
    endpoint = 'postProjectInternal'
    return 'http://{}:{}/{}'.format(address, port, endpoint)


def get_task_uri(port):
    """ gets endpoint for task creation given the port """
    address = '127.0.0.1'
    endpoint = 'postTasks'
    return 'http://{}:{}/{}'.format(address, port, endpoint)


@pytest.mark.usefixtures(SERVER_FIXTURE_NAME)
def test_create_project(get_config):
    """ test project creation internal API """
    port = get_config['port']
    uri = get_create_uri(port)
    body = {
        'fields': get_sample_fields('internal_project'),
        'files': {
            'item_file': 'examples/image_list.yml'
        }
    }
    response = requests.post(uri, data=json.dumps(
        body), timeout=1, headers=HEADERS)
    assert response.status_code == 200

    # test repeated name fails
    response = requests.post(uri, data=json.dumps(
        body), timeout=1, headers=HEADERS)
    assert response.status_code == 400


@pytest.mark.usefixtures(SERVER_FIXTURE_NAME)
def test_create_project_no_items(get_config):
    """ test internal project creation API allows adding items later """
    port = get_config['port']
    uri = get_create_uri(port)
    project_name = 'other_project'
    body = {
        'fields': get_sample_fields(project_name),
        'files': {}
    }
    response = requests.post(uri, data=json.dumps(
        body), timeout=1, headers=HEADERS)
    assert response.status_code == 200

    # now add the items
    task_uri = get_task_uri(port)
    body = {
        'projectName': project_name,
        'items': 'examples/image_list.yml'
    }
    response = requests.post(task_uri, data=json.dumps(body),
                             timeout=1, headers=HEADERS)
    assert response.status_code == 200
