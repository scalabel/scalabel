from tqdm import tqdm
import numpy as np

from scalabel_bot.task.dd3d import DD3D


class DD3DTraining:
    def __init__(self) -> None:
        self.dd3d = DD3D()

    def import_data(self, task):
        data = self.dd3d.import_data(task)
        return data

    def import_model(self, device=None):
        self.dd3d.import_model(device, "training")

    def import_func(self):
        def train(task, data):
            # TODO: implement DD3D finetuning
            pass

        return train

    def import_task(self, device):
        self.import_model(device)
        func = self.import_func()
        return func


if __name__ == "__main__":
    dd3d = DD3DTraining()
