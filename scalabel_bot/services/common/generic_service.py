from abc import ABC, abstractmethod
from typing import Callable, List
from tqdm import tqdm

from scalabel_bot.common.consts import ServiceMode, T
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.services.common.generic_model import GenericModel


class GenericService(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self._name: str = ""
        self._service_mode: ServiceMode = ServiceMode.NONE
        self._model: GenericModel
        self._func: Callable[[TaskMessage, List[T]], List[T]]

    def import_data(self, task: TaskMessage) -> List[T]:
        data: List[T] = self._model.import_data(task)
        return data

    def import_model(
        self, device: int = None
    ) -> Callable[[TaskMessage, List[T]], List[T]]:
        self._model.import_model(device, self._service_mode)
        return self._func

    @abstractmethod
    def import_func(self) -> Callable[[TaskMessage, List[T]], List[T]]:
        pass
