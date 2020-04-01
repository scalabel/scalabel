import requests
import json


class ScalabelAPI():
    """ Internal API for scalabel """

    def __init__(self, port):
        self.headers = {'Content-Type': 'application/json'}
        self.port = port
        self.address = '127.0.0.1'
        self.create_project_endpoint = 'postProjectInternal'
        self.add_tasks_endpoint = 'postTasks'
        self.default_items = 'examples/image_list.yml'

    def make_uri(self, endpoint):
        """ Gets uri for the given endpoint """
        return 'http://{}:{}/{}'.format(self.address, self.port, endpoint)

    def create_project(self, fields_dict, files_dict):
        """ Project creation with full range of arguments allowed """
        body = {
            'fields': fields_dict,
            'files': files_dict
        }
        uri = self.make_uri(self.create_project_endpoint)
        response = requests.post(uri, data=json.dumps(
            body), timeout=1, headers=self.headers)
        return response

    def add_default_tasks(self, project_name):
        """ Adds tasks to an existing project """
        body = {
            'projectName': project_name,
            'items': self.default_items
        }
        uri = self.make_uri(self.add_tasks_endpoint)
        response = requests.post(uri, data=json.dumps(
            body), timeout=1, headers=self.headers)
        return response

    def create_default_project(self, project_name, use_items=True):
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
