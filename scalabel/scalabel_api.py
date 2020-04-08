""" Scalabel API """
import json
import requests
from urllib3.exceptions import HTTPError


class LowLevelAPI():
    """ Low level api for making requests """

    def __init__(self, port: int) -> None:
        self.headers = {'Content-Type': 'application/json'}
        self.port = port
        self.address = '127.0.0.1'
        self.create_project_endpoint = 'postProjectInternal'
        self.add_tasks_endpoint = 'postTasks'

    def make_uri(self, endpoint: str) -> str:
        """ Gets uri for the given endpoint """
        return 'http://{}:{}/{}'.format(self.address, self.port, endpoint)

    def get_create_project_uri(self) -> str:
        """ Gets uri for creating projects """
        return self.make_uri(self.create_project_endpoint)

    def get_add_tasks_uri(self) -> str:
        """ Gets uri for adding tasks """
        return self.make_uri(self.add_tasks_endpoint)

    def make_request(self, uri: str, body: dict) -> None:
        """ Makes the request to the node js server """
        response = requests.post(uri, data=json.dumps(
            body), timeout=1, headers=self.headers)
        code = response.status_code
        if code != 200:
            raise HTTPError(
                'API call failed with status code {}'.format(code))


class ScalabelProject():
    """ Representation of a scalabel project created via the API """

    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.api = LowLevelAPI(self.port)
        self.default_items = 'examples/image_list.yml'

    def add_default_tasks(self) -> None:
        """ Add default item list to the project """
        self.add_tasks(self.default_items)

    def add_tasks(self, items: str) -> None:
        """ Adds tasks to the project """
        body = {
            'projectName': self.name,
            'items': items
        }
        uri = self.api.get_add_tasks_uri()
        self.api.make_request(uri, body)


class ScalabelAPI():
    """ Internal API for scalabel """

    def __init__(self, port: int) -> None:
        self.port = port
        self.api = LowLevelAPI(self.port)
        self.default_items = 'examples/image_list.yml'

    def create_project(self, fields: dict, files: dict) -> ScalabelProject:
        """ Project creation with full range of arguments allowed """
        body = {
            'fields': fields,
            'files': files
        }
        uri = self.api.get_create_project_uri()
        print(uri)
        self.api.make_request(uri, body)
        return ScalabelProject(fields['project_name'], self.port)

    def create_default_project(
            self, project_name: str, use_items=True) -> ScalabelProject:
        """ Example usage for minimal project """
        fields = {
            'project_name': project_name,
            'item_type': 'image',
            'label_type': 'box2d',
            'task_size': 10
        }
        files = {}
        if use_items:
            files['item_file'] = self.default_items

        return self.create_project(fields, files)
