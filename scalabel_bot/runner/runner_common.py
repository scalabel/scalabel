from scalabel_bot.common.consts import Timers
from scalabel_bot.profiling.timer import timer


class ModelSummary:
    def __init__(self, mode, devices, model_name, model_class):
        self._mode = mode
        self._devices = devices
        self._model_name = model_name
        self._model_class = model_class
        self._func = None

    @timer(Timers.PERF_COUNTER)
    def execute(self, task, data):
        return self._func(task, data)

    @timer(Timers.PERF_COUNTER)
    def load_model(self):
        self._func = self._model_class().import_task(self._devices)

    @timer(Timers.PERF_COUNTER)
    def load_data(self, task):
        return self._model_class().import_data(task)
