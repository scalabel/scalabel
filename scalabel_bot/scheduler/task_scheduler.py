from typing import Dict, List, Tuple
from queue import Queue
from multiprocessing import Event, Process

from scalabel_bot.common.consts import State, Timers
from scalabel_bot.common.func import cantor_pairing
from scalabel.common.logger import logger
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.profiling.timer import timer
from scalabel_bot.scheduler.scheduling_policy import SJFLBAP


class TaskScheduler(Process):
    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        stop_run: Event,
        num_runners: int,
        runner_status: Dict[int, State],
        runner_status_queue: Queue[Tuple[int, int, State]],
        runner_ect: Dict[int, int],
        runner_ect_queue: Queue[Tuple[int, int, int]],
        requests_queue: List[TaskMessage],
        clients: Dict[str, int],
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop_run: Event = stop_run
        self._num_runners: int = num_runners
        self._runner_status: Dict[int, State] = runner_status
        self._runner_status_queue: Queue[
            Tuple[int, int, State]
        ] = runner_status_queue
        self._runner_ect_queue: Queue[Tuple[int, int, int]] = runner_ect_queue
        self._runner_ect: Dict[int, int] = runner_ect
        self._runner_count: int = -1
        self._requests_queue: List[TaskMessage] = requests_queue
        self._clients: Dict[str, int] = clients
        self._policy: SJFLBAP = SJFLBAP(self._num_runners)

    def run(self) -> None:
        try:
            while not self._stop_run.is_set():
                device, runner_id, status = self._runner_status_queue.get()
                logger.debug(
                    f"{self._name}: Runner {device}-{runner_id} {status}"
                )
                self._runner_status[cantor_pairing(device, runner_id)] = status
        except KeyboardInterrupt:
            return

    def _update_ect(self):
        try:
            while not self._stop_run.is_set():
                device, runner_id, ect = self._runner_ect_queue.get()
                self._runner_ect[cantor_pairing(device, runner_id)] = ect
        except KeyboardInterrupt:
            return

    @timer(Timers.THREAD_TIMER)
    def schedule(self) -> Tuple[TaskMessage, int]:
        while not self._runner_ect_queue.empty():
            device, runner_id, ect = self._runner_ect_queue.get()
            self._runner_ect[cantor_pairing(device, runner_id)] = ect
        while self._requests_queue:
            task = self._policy.choose_task(self._requests_queue)
            if task["clientId"] in self._clients.keys():
                break
        resource = self._policy.choose_resource(self._runner_ect)
        return task, resource

    @property
    def runner_status(self) -> Dict[int, State]:
        return self._runner_status
