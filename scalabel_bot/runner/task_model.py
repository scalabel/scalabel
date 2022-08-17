from typing import Callable, List

from scalabel_bot.common.consts import Timers
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.profiling.timer import timer


class TaskModel:
    def __init__(self, mode, devices, model_name, model_class) -> None:
        self._mode: str = mode
        self._devices: int = devices
        self._model_name: str = model_name
        self._model_class: object = model_class
        self._func: Callable[[TaskMessage, List[object]], List[object]] = None

    @timer(Timers.PERF_COUNTER)
    def execute(self, task, data) -> List[object]:
        output: List[object] = self._func(task, data)
        return output

    @timer(Timers.PERF_COUNTER)
    def load_model(self) -> None:
        self._func = self._model_class().import_task(self._devices)

    @timer(Timers.PERF_COUNTER)
    def load_data(self, task) -> List[object]:
        return self._model_class().import_data(task)
