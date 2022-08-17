from abc import ABC, abstractmethod
from typing import Dict, Tuple  # pylint: disable=unused-import
from torch.multiprocessing import (  # pylint: disable=unused-import
    Event,
    Process,
    Queue,
)

from scalabel_bot.common.consts import State, Timers
from scalabel_bot.common.func import cantor_pairing
from scalabel.common.logger import logger
from scalabel_bot.profiling.timer import timer


class SchedulingPolicy(ABC):
    @abstractmethod
    def select_next(
        self, runner_ect: Dict[int, int], runner_id: int = None
    ) -> int:
        pass


class RoundRobin(SchedulingPolicy):
    @timer(Timers.THREAD_TIMER)
    def select_next(self, runner_ect: Dict[int, int], runner_id: int) -> int:
        next_runner_id = (runner_id + 1) % len(runner_ect)
        return list(runner_ect.keys())[next_runner_id]


class LoadBalancing(SchedulingPolicy):
    @timer(Timers.THREAD_TIMER)
    def select_next(
        self, runner_ect: Dict[int, int], runner_id: int = None
    ) -> int:
        next_runner_id = min(runner_ect, key=runner_ect.get)
        return next_runner_id


class TaskScheduler(Process):
    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        runner_status: Dict[int, State],
        runner_status_queue: "Queue[Tuple[int, int, State]]",
        runner_ect: Dict[int, int],
        runner_ect_queue: "Queue[Tuple[int, int, int]]",
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop_run: Event = Event()
        self._policy: SchedulingPolicy = LoadBalancing()
        self._runner_status: Dict[int, State] = runner_status
        self._runner_status_queue: "Queue[Tuple[int, int, State]]" = (
            runner_status_queue
        )
        self._runner_ect_queue: "Queue[Tuple[int, int, int]]" = (
            runner_ect_queue
        )
        self._runner_ect: Dict[int, int] = runner_ect
        self._curr_runner_id: int = 0

    def run(self) -> None:
        while not self._stop_run.is_set():
            device, runner_id, status = self._runner_status_queue.get()
            logger.debug(f"{self._name}: Runner {device}-{runner_id} {status}")
            self._runner_status[cantor_pairing(device, runner_id)] = status

    def _update_ect(self):
        while not self._stop_run.is_set():
            device, runner_id, ect = self._runner_ect_queue.get()
            self._runner_ect[cantor_pairing(device, runner_id)] = ect

    @timer(Timers.THREAD_TIMER)
    def schedule(self) -> int:
        while not self._runner_ect_queue.empty():
            device, runner_id, ect = self._runner_ect_queue.get()
            self._runner_ect[cantor_pairing(device, runner_id)] = ect
        if isinstance(self._policy, LoadBalancing):
            self._curr_runner_id = self._policy.select_next(self._runner_ect)
        elif isinstance(self._policy, RoundRobin):
            self._curr_runner_id = self._policy.select_next(
                self._runner_ect, self._curr_runner_id
            )
        return self._curr_runner_id

    @timer(Timers.THREAD_TIMER)
    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name}: stopping...")
        self._stop_run.set()
        logger.debug(f"{self._name}: stopped!")

    @property
    def runner_status(self) -> Dict[int, State]:
        return self._runner_status
