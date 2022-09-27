# -*- coding: utf-8 -*-
"""Scalabel Bot Runner Service Model.

This module exposes a common API for runners to run tasks.

Todo:
    * None
"""


from typing import Callable, List, TypeVar

from scalabel_bot.common.consts import Timers
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.profiling.timer import timer
from scalabel_bot.services.common.generic_service import GenericService

T = TypeVar("T")


class ServiceModel:
    def __init__(
        self,
        mode: str,
        devices: int,
        service_name: str,
        service: GenericService,
    ) -> None:
        self._mode: str = mode
        self._devices: int = devices
        self._service_name: str = service_name
        self._service: GenericService = service
        self._func: Callable[[TaskMessage, List[object]], List[object]]

    @timer(Timers.PERF_COUNTER)
    def execute(self, task, data) -> List[object]:
        output: List[object] = self._func(task, data)
        return output

    @timer(Timers.PERF_COUNTER)
    def load_model(self) -> None:
        self._func = self._service.import_model(self._devices)

    @timer(Timers.PERF_COUNTER)
    def load_data(self, task: TaskMessage) -> List[T]:
        return self._service.import_data(task)
