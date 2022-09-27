from abc import ABC, abstractmethod
from redis import Redis
from typing import List

from scalabel_bot.common.consts import REDIS_HOST, REDIS_PORT, ServiceMode, T
from scalabel_bot.common.message import TaskMessage


class GenericModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self._name: str = ""
        self._data_loader = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            encoding="utf-8",
            decode_responses=True,
        )

    @abstractmethod
    def import_data(self, task: TaskMessage) -> List[T]:
        pass

    @abstractmethod
    def import_model(
        self, device: int = None, service_mode: ServiceMode = ServiceMode.NONE
    ) -> None:
        pass

    @abstractmethod
    def __call__(self, *args) -> List[T]:
        pass
