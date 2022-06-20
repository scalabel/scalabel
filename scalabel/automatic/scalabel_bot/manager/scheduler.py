from time import sleep
from abc import ABC, abstractmethod
from typing import List, OrderedDict, Tuple  # pylint: disable=unused-import
from torch.multiprocessing import (  # pylint: disable=unused-import
    Event,
    Process,
    Queue,
)

from scalabel.automatic.scalabel_bot.common.consts import State, Timers
from scalabel.automatic.scalabel_bot.common.logger import logger
from scalabel.automatic.scalabel_bot.profiling.timer import timer


class Policy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_next(self, runners_list: List[int]) -> int:
        pass


class RoundRobinPolicy(Policy):
    @timer(Timers.THREAD_TIMER)
    def select_next(self, runners_list: List[int]) -> int:
        return runners_list[0]


class Scheduler(Process):
    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        num_runners: List[int],
        runner_status: OrderedDict[int, State],
        runner_status_queue: "Queue[Tuple[int, State]]",
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop_run: Event = Event()
        self._policy: Policy = RoundRobinPolicy()
        self._runner_idx: List[int] = num_runners
        self._runner_status: OrderedDict[int, State] = runner_status
        self._runner_status_queue: "Queue[Tuple[int, State]]" = (
            runner_status_queue
        )

    def run(self) -> None:
        while not self._stop_run.is_set():
            runner_id, status = self._runner_status_queue.get()
            logger.debug(f"{self._name}: Runner {runner_id} {status}")
            self._runner_status[runner_id] = status

    @timer(Timers.THREAD_TIMER)
    def schedule(self) -> int:
        free_runners = self._get_free_runners()
        while len(free_runners) < 1:
            sleep(0.001)
            free_runners = self._get_free_runners()

        next_available_runner: int = self._policy.select_next(free_runners)
        self.runner_status[next_available_runner] = State.RESERVED
        logger.debug(
            f"{self._name}: Next available runner is {next_available_runner}"
        )
        return next_available_runner

    def _get_free_runners(self) -> List[int]:
        return [
            runner_id
            for runner_id, status in self.runner_status.items()
            if status == State.IDLE
        ]

    @timer(Timers.THREAD_TIMER)
    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name}: stopping...")
        self._stop_run.set()
        logger.debug(f"{self._name}: stopped!")

    @property
    def runner_idx(self) -> List[int]:
        return self._runner_idx

    @property
    def runner_status(self) -> OrderedDict[int, State]:
        return self._runner_status
