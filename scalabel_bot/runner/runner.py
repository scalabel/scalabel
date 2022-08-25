# -*- coding: utf-8 -*-
"""Scalabel Bot Runner.

This module executes task requests from clients.

Todo:
    * Efficient model switching
    * Unload model cache and data cache upon switching models
    * Restart runner when there are any faults
    * Catch runtime errors of tasks
"""


from time import sleep
from typing import Dict, List, Tuple, Type
import torch
from queue import Queue
from multiprocessing import Event, Pipe, Process
from multiprocessing.synchronize import Event as EventClass
from threading import Thread
import gc

from scalabel_bot.common.consts import ESTCT, MODELS, State, Timers
from scalabel.common.logger import logger
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.profiling.timer import timer
from scalabel_bot.runner.service_model import ServiceModel
from scalabel_bot.services.common.generic_service import GenericService


class Runner(Process):
    """Runner process.

    It receives tasks from the manager
    and allocates them to the available workers.
    It collects results from the workers and sends them back to the manager.

    Attributes:
        device (int): ID of the GPU that the runner is running on.
        model_list (List[str]): List of model names.

        worker_list
            (List[
                Tuple[
                    connection.Connection,
                    WorkerProc,
                    connection.Connection,
                    connection.Connection
                ]
            ]): List of worker processes.

        worker_status (OrderedDict[int, WorkerStatus]):
            Dictionary containing the status of the workers
        cur_w_idx (int): Index of the current worker.
        comms_server (RedisServer):
            Redis server for receiving status updates from the workers
            and updating own status to the manager.
        task_server (RedisServer):
            Redis server for receiving tasks from the manager
            and sending results to the manager.
    """

    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        mode: str,
        device: int,
        runner_id: int,
        runner_status_queue: Queue[Tuple[int, int, State]],
        runner_ect_queue: Queue[Tuple[int, int, int]],
        services_classes: Dict[str, Type[GenericService]],
        results_queue: Queue[TaskMessage],
        clients: Dict[str, int],
    ) -> None:
        super().__init__()
        self._name = self.__class__.__name__
        self._stop_run: EventClass = Event()
        self._mode: str = mode
        self._device: int = device
        self._id: int = runner_id
        self._status: State = State.STARTUP
        self._runner_status_queue: Queue[
            Tuple[int, int, State]
        ] = runner_status_queue
        self._runner_ect_queue: Queue[Tuple[int, int, int]] = runner_ect_queue
        self._task_in, self._task_out = Pipe()
        self._task_queue: List[TaskMessage] = []
        self._results_queue: Queue[TaskMessage] = results_queue
        self._clients: Dict[str, int] = clients
        self._services_classes: Dict[
            str, Type[GenericService]
        ] = services_classes
        self._services_models: Dict[str, ServiceModel] = {}
        self._max_retries: int = 2
        self._ect: int = 0

    def run(self) -> None:
        """Main runner function that sets up the runner and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        try:
            logger.debug(f"{self._name} {self._device}-{self._id}: start")
            if self._mode == "gpu":
                logger.debug(
                    f"{self._name} {self._device}-{self._id}: share GPU memory"
                )
                self._load_jobs: List[Thread] = []
                for service_name, service in self._services_classes.items():
                    service_model: ServiceModel = ServiceModel(
                        mode=self._mode,
                        devices=self._device,
                        service_name=service_name,
                        service=service(),
                    )
                    load_model = Thread(target=service_model.load_model)
                    load_model.daemon = True
                    load_model.start()
                    load_model.join()
                    self._load_jobs.append(load_model)
                    self._services_models[service_name] = service_model
                    # break
                logger.debug(
                    f"{self._name} {self._device}-{self._id}: import models"
                )
            recv_task = Thread(target=self._recv_task)
            recv_task.daemon = True
            recv_task.start()
            self._update_ect()
            self._update_status(State.IDLE)

            while not self._stop_run.is_set():
                if not self._task_queue:
                    continue
                task: TaskMessage = self._task_queue.pop(0)
                if task["clientId"] in self._clients:
                    self._manage_task(task)
                self._update_ect("remove", task)
        except KeyboardInterrupt:
            self._stop_run.set()
            return

    @timer(Timers.THREAD_TIMER)
    def _update_status(self, status: State) -> None:
        """Updates own runner status based on worker statuses."""
        try:
            self._status = status
            logger.debug(
                f"{self._name} {self._device}-{self._id}: Updating status to"
                f" {self._status}"
            )
            self._runner_status_queue.put(
                (self._device, self._id, self._status)
            )
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _calc_ect(self, task: TaskMessage) -> int:
        stct: int = ESTCT[task["mode"]][MODELS[task["taskType"]]]
        return stct * task["dataSize"]

    def _update_ect(self, mode: str = "", task: TaskMessage = None) -> None:
        if task:
            if mode == "append":
                self._ect += self._calc_ect(task)
            elif mode == "remove":
                self._ect -= self._calc_ect(task)
        self._runner_ect_queue.put((self._device, self._id, self._ect))

    def _recv_task(self) -> None:
        try:
            while not self._stop_run.is_set():
                task: TaskMessage = self._task_out.recv()
                print(self._clients.items())
                if task["clientId"] in self._clients:
                    self._task_queue.append(task)
                    self._update_ect("append", task)
        except KeyboardInterrupt:
            return

    @timer(Timers.PERF_COUNTER)
    def _manage_task(self, task: TaskMessage) -> None:
        if self._mode == "gpu":
            service_model: ServiceModel = self._services_models[
                f"{MODELS[str(task['taskType'])]}_{task['mode']}"
            ]
            if task["items"]:
                output: object | None = self._execute_task(task, service_model)
            else:
                return
        else:
            logger.debug(
                f"{self._name} {self._device}-{self._id}: CPU debug"
                " mode task execution"
            )
            output = {
                "boxes": [
                    [700.7190, 69.1574, 720.9227, 134.3829],
                    [730.7932, 224.3562, 749.0588, 284.2728],
                    [1169.0742, 0.0000, 1195.4609, 44.7009],
                    [347.8973, 140.9991, 426.8250, 225.2764],
                ],
                "scores": [0.8418, 0.7967, 0.7857, 0.7330],
                "pred_classes": [9, 9, 9, 11],
            }
            sleep(task["ect"] / 1000)
        if output is not None:
            task["status"] = State.SUCCESS.value
        else:
            task["status"] = State.FAILED.value
        task["output"] = output
        self._results_queue.put(task)
        # with torch.no_grad():
        #     gc.collect()
        #     torch.cuda.empty_cache()

    def _execute_task(
        self, task: TaskMessage, service_model: ServiceModel
    ) -> object | None:
        """Executes a task."""
        data: List[object] = []
        output: List[object] = []
        for attempt in range(self._max_retries):
            if attempt > 0:
                logger.info(f"Retrying task #{attempt}")
            try:
                if not data:
                    data = service_model.load_data(task)
            except RuntimeError as runtime_err:
                logger.error(runtime_err)
                logger.error(
                    f"{self._name} {self._device}-{self._id}: data loading"
                    " failed!"
                )
                continue
            try:
                if not output:
                    output = service_model.execute(task, data)
                    return output
            except RuntimeError as runtime_err:
                logger.error(runtime_err)
                logger.error(
                    f"{self._name} {self._device}-{self._id}: task failed!"
                )
                continue
        logger.error(
            f"{self._name} {self._device}-{self._id}: max retries exceeded, "
            "skipping task"
        )
        return None

    @property
    def task_in(self):
        return self._task_in

    @property
    def task_out(self):
        return self._task_out

    @property
    def ect(self):
        return self._ect
