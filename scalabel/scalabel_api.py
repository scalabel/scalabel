""" Scalabel API """
import json
import requests
from urllib3.exceptions import HTTPError


class ScalabelAPI():
    """ Internal API for scalabel """

    def __init__(self, port: int) -> None:
        self.headers = {'Content-Type': 'application/json'}
        self.port = port
        self.address = '127.0.0.1'
        self.create_project_endpoint = 'postProjectInternal'
        self.add_tasks_endpoint = 'postTasks'
        self.default_items = 'examples/image_list.yml'

    def make_uri(self, endpoint: str) -> str:
        """ Gets uri for the given endpoint """
        return 'http://{}:{}/{}'.format(self.address, self.port, endpoint)

    def make_request(self, uri: str, body: dict) -> None:
        """ Makes the request to the node js server """
        response = requests.post(uri, data=json.dumps(
            body), timeout=1, headers=self.headers)
        if response.status_code != 200:
            raise HTTPError('API call failed')

    def create_project(self, fields: dict, files: dict) -> ScalabelProject:
        """ Project creation with full range of arguments allowed """
        body = {
            'fields': fields,
            'files': files
        }
        uri = self.make_uri(self.create_project_endpoint)
        self.make_request(uri, body)
        return ScalabelProject(fields['project_name'], self)

    def add_default_tasks(self, project_name: str) -> None:
        """ Adds default item list to an existing project """
        self.add_tasks(project_name, self.default_items)

    def add_tasks(self, project_name: str, items: str) -> None:
        """ Adds tasks to an existing project """
        body = {
            'projectName': project_name,
            'items': items
        }
        uri = self.make_uri(self.add_tasks_endpoint)
        self.make_request(uri, body)

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


class ScalabelProject():
    """ Representation of a scalabel project created via the API """

    def __init__(self, name: str, api: ScalabelAPI):
        self.name = name
        self.api = api

    def add_default_tasks(self) -> None:
        """ Add default item list to the project """
        self.api.add_default_tasks(self.name)

    def add_tasks(self, items: str) -> None:
        """ Add tasks to the project """
        self.api.add_tasks(self.name, items)
