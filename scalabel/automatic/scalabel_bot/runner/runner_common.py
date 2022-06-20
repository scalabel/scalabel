from scalabel.automatic.scalabel_bot.common.consts import Timers
from scalabel.automatic.scalabel_bot.profiling.timer import timer


class ModelSummary:
    def __init__(self, mode, device, model_name, model_class):
        """ """
        self.mode = mode
        self.device = device
        self.model_name = model_name
        self.model_class = model_class

    @timer(Timers.THREAD_TIMER)
    def execute(self, task, data):
        return self.func(task, data)

    @timer(Timers.THREAD_TIMER)
    def load_model(self):
        (
            self.model,
            self.func,
        ) = self.model_class().import_task(self.device)

    @timer(Timers.THREAD_TIMER)
    def load_data(self, task):
        return self.model_class().import_data(task)
