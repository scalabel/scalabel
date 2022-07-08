from tqdm import tqdm
from pprint import pprint

from scalabel_bot.common.consts import Timers
from scalabel_bot.profiling.timer import timer
from scalabel_bot.task.fsdet import FSDET


TASK_NAME = "fsdet_inference"


class FSDETInference:
    def __init__(self) -> None:
        self.fsdet = FSDET()

    def import_data(self, task):
        data = self.fsdet.import_data(task)
        return data

    def import_model(self, device=None):
        self.predictor = self.fsdet.import_model(device)

    def import_func(self):
        def inference(task, data):
            output = []
            for img in tqdm(
                data,
                desc=f"Task {task['taskId']}",
                leave=True,
                position=0,
                unit="items",
            ):
                predictions = self.predictor(img)
                instances = predictions["instances"]
                boxes = []
                for pred_box in instances.pred_boxes:
                    boxes.append(pred_box.cpu().numpy().tolist())
                output.append({"boxes": boxes})
            return output

        return inference

    @timer(Timers.THREAD_TIMER)
    def import_task(self, device):
        self.import_model(device)
        func = self.import_func()
        return func


MODEL_CLASS = FSDETInference
