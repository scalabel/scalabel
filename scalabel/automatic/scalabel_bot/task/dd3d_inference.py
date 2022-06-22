from tqdm import tqdm

from scalabel.automatic.scalabel_bot.common.consts import Timers
from scalabel.automatic.scalabel_bot.profiling.timer import timer
from scalabel.automatic.scalabel_bot.task.dd3d import DD3D


TASK_NAME = "dd3d_inference"


class DD3DInference:
    def __init__(self) -> None:
        self.dd3d = DD3D()

    def import_data(self, task):
        data = self.dd3d.import_data(task)
        return data

    def import_model(self, device=None):
        self.dd3d.import_model(device)

    def import_func(self):
        def inference(task, data):
            output = [
                self.dd3d([img])
                for img in tqdm(
                    data,
                    desc=f"Task {task['projectName']}_{task['taskId']}",
                    leave=True,
                    position=0,
                    unit="items",
                )
            ]
            return output

        return inference

    @timer(Timers.THREAD_TIMER)
    def import_task(self, device):
        self.import_model(device)
        func = self.import_func()
        return func


MODEL_CLASS = DD3DInference
