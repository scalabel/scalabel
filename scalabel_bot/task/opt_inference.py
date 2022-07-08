from tqdm import tqdm

from scalabel_bot.task.opt import OPT


TASK_NAME = "opt_inference"


class OPTInference:
    def __init__(self) -> None:
        self.opt = OPT()

    def import_data(self, task):
        data = self.opt.import_data(task)
        return data

    def import_model(self, device=None):
        self.opt.import_model(device)

    def import_func(self):
        def inference(task, data):
            output = []
            for prompt, length in tqdm(
                data,
                desc=f"Task {task['projectName']}_{task['taskId']}",
                leave=True,
                position=0,
                unit="items",
            ):
                output.extend(self.opt(prompt, length))
            return output

        return inference

    def import_task(self, device):
        self.import_model(device)
        func = self.import_func()
        return func


MODEL_CLASS = OPTInference
