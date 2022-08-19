# -*- coding: utf-8 -*-
"""Scalabel Bot Manager.

This module creates and manages various runners and schedulers.

Todo:
    * Dynamic allocation of GPUs to runners.
"""


from time import sleep
import os
import importlib
from types import ModuleType
from typing import List, Dict, Tuple, Type
from threading import Thread
from pprint import pformat
import json
from queue import Queue
from multiprocessing import Manager, Queue as MPQueue
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event as EventClass

from scalabel_bot.common.consts import (
    MODEL_LIST_FILE,
    REDIS_HOST,
    REDIS_PORT,
    State,
    Timers,
)
from scalabel_bot.common.func import cantor_pairing
from scalabel.common.logger import logger
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.manager.gpu_resource_allocator import GPUResourceAllocator
from scalabel_bot.scheduler.task_scheduler import TaskScheduler
from scalabel_bot.manager.client_manager import ClientManager
from scalabel_bot.profiling.timer import timer
from scalabel_bot.runner.runner import Runner
from scalabel_bot.server.stream import ManagerRequestsStream
from scalabel_bot.server.pubsub import ManagerRequestsPubSub
from scalabel_bot.services.common.generic_service import GenericService


class BotManager:
    """Manager class that acts as a middleman between clients and runners.

    It has two main functions:
        1. It receives task requests from clients and sends them
           to the task scheduler to be allocated to runners.
        2. It receives task results from runners and sends them
           to the appropriate client.

    Attributes:
        _name (`str`): Name of the class.
        _stop_run (`EventClass`): Global run flag.
        _mode (`str`): Mode of the manager (CPU or GPU).

        _num_gpus (`int`): Number of GPUs specified by the user.
        _gpu_ids (`List[int]`): List of GPU IDs specified by the user.
            Defaults to an empty list, which means all available GPUs are used.
        _allocated_gpus (`List[int]`): List of GPUs that have been
            allocated to runners.
        _gra (`GPUResourceAllocator`):
            Object that polls for available GPUs and reserves them.

        _manager (`SyncManager`): Multiprocessing manager object that manages
            shared data structures across processes.



        _runner_status (`Dict[int, State]`): Dictionary of runner IDs and
            their current status.
        _runner_status_queue (`Queue[Tuple[int, int, State]]`): Queue for
            runners to update their status.
        _runner_ect (`Dict[int, int]`): Dictionary of runner IDs and their
            overall estimated task completion time.
        _runner_ect_queue (`Queue[Tuple[int, int, int]]`): Queue for runners to
            update their overall estimated task completion time.
        _runners (`Dict[int, Runner]`): Dictionary of runner IDs and
            runner processes.

        _requests_queue (`List[TaskMessage]`): Queue where incoming task
            requests are stored.
        _results_queue (`Queue[TaskMessage]`): Queue where incoming task
            results are stored.
        _req_stream (`ManagerRequestsStream`): Redis stream thread that
            1. Receives task requests and stores them in the requests queue.
            2. Sends task results to the appropriate client.
        _req_pubsub (`ManagerRequestsPubSub`): Redis pubsub thread that
            1. Receives task requests and stores them in the requests queue.
            2. Sends task results to the appropriate client.

        _clients (`Dict[str, int]`): Dictionary of client IDs and their TTLs.
        _client_manager (`ClientManager`): Client manager process that
            manages client connections.

        _services_list (`List[str]`): List of model services to be loaded
            into memory.
        _services_classes (`Dict[str, object]`): Dictionary of model services
            names and their object classes.

        _task_scheduler (`TaskScheduler`): Process that manages task
            and runner allocation.
        _results_checker (`Thread`): Thread that polls for
            task results from runners.

        _tasks_complete (`int`): Number of tasks that have been completed.
        _tasks_failed (`int`): Number of tasks that have failed.
    """

    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        stop_run: EventClass,
        mode: str = "gpu",
        num_gpus: int = 0,
        gpu_ids: List[int] = None,
    ) -> None:
        """Initializes the BotManager class.

        Args:
            stop_run (`EventClass`): Global run flag.
            mode (`str`, optional): Mode of the manager (CPU or GPU).
                Defaults to "gpu".
            num_gpus (`int`, optional): Number of GPUs specified by the user.
                Defaults to 0, which means all available GPUs are used.
            gpu_ids (`List[int]`, optional): List of GPU IDs specified by the
                user. Defaults to None, which means all available GPUs are used.
        """
        super().__init__()

        self._name: str = self.__class__.__name__
        self._stop_run: EventClass = stop_run
        self._mode: str = mode

        self._num_gpus: int = num_gpus
        self._gpu_ids: List[int] = gpu_ids if gpu_ids else []
        self._allocated_gpus: List[int] = []
        if self._mode == "gpu":
            self._gra: GPUResourceAllocator = GPUResourceAllocator()
            self._allocated_gpus = self._gra.reserve_gpus(
                self._num_gpus, self._gpu_ids
            )
            logger.info(f"{self._name}: Allocated GPUs {self._allocated_gpus}")
            self._gra.warmup_gpus(gpus=self._allocated_gpus)
        else:
            self._allocated_gpus = [*range(self._num_gpus)]

        self._manager: SyncManager = Manager()

        self._runner_status: Dict[int, State] = self._manager.dict()
        self._runner_status_queue: Queue[Tuple[int, int, State]] = MPQueue()
        self._runner_ect: Dict[int, int] = self._manager.dict()
        self._runner_ect_queue: Queue[Tuple[int, int, int]] = MPQueue()
        self._runners: Dict[int, Runner] = {}

        self._requests_queue: List[TaskMessage] = self._manager.list()
        self._results_queue: Queue[TaskMessage] = MPQueue()

        self._requests_stream: ManagerRequestsStream = ManagerRequestsStream(
            stop_run=self._stop_run,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._requests_queue,
        )
        self._requests_pubsub: ManagerRequestsPubSub = ManagerRequestsPubSub(
            stop_run=self._stop_run,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._requests_queue,
        )

        self._clients: Dict[str, int] = self._manager.dict()
        self._client_manager: ClientManager = ClientManager(
            stop_run=self._stop_run,
            clients=self._clients,
            requests_channel=self._requests_stream.sub_stream,
        )

        self._services_list: List[str] = []
        self._services_classes: Dict[
            str, Type[GenericService]
        ] = self._manager.dict()

        self._task_scheduler: TaskScheduler = TaskScheduler(
            stop_run=self._stop_run,
            num_runners=len(self._allocated_gpus),
            runner_status=self._runner_status,
            runner_status_queue=self._runner_status_queue,
            runner_ect=self._runner_ect,
            runner_ect_queue=self._runner_ect_queue,
            requests_queue=self._requests_queue,
            clients=self._clients,
        )

        self._results_checker: Thread = Thread(
            target=self._check_result,
            args=(
                self._results_queue,
                self._requests_stream,
                self._requests_pubsub,
            ),
        )

        self._tasks_complete: int = 0
        self._tasks_failed: int = 0

    def run(self) -> None:
        """Sets up the bot manager and runs it."""
        self._client_manager.daemon = True
        self._client_manager.start()

        self._requests_stream.daemon = True
        self._requests_stream.start()

        self._requests_pubsub.daemon = True
        self._requests_pubsub.start()

        logger.info(  # type: ignore
            "\n*********************************************\n"
            f"{self._name}: Ready to receive requests!\n"
            "*********************************************"
        )

        self._load_models()

        self._task_scheduler.daemon = True
        self._task_scheduler.start()

        self._create_runners()

        self._results_checker.daemon = True
        self._results_checker.start()

        while (len(self._runner_status) < len(self._runners)) and (
            len(self._runner_ect) < len(self._runners)
        ):
            sleep(1)
        logger.info(  # type: ignore
            "\n******************************************\n"
            f"{self._name}: Ready to execute tasks!\n"
            "******************************************"
        )

        try:
            while not self._stop_run.is_set():
                if self._requests_queue:
                    self._allocate_task()
        except KeyboardInterrupt:
            self._stop_run.set()
            self._requests_stream.shutdown()

    @timer(Timers.PERF_COUNTER)
    def _load_services_list(self, file_name: str) -> None:
        """Loads a list of model services to be used by the runners.

        Args:
            file_name (`str`): Path to the file containing the list of
                model services.
        """
        if not os.path.exists(file_name):
            logger.error(
                f"{self._name}: Model services list {file_name} not found"
            )
            return

        with open(file=file_name, mode="r", encoding="utf-8") as f:
            self._services_list = [line.strip() for line in f.readlines()]

    @timer(Timers.PERF_COUNTER)
    def _load_models(self) -> None:
        """Loads the model classes to be used by the runners."""
        self._load_services_list(file_name=MODEL_LIST_FILE)
        for service_name in self._services_list:
            service_module: ModuleType = importlib.import_module(
                "scalabel_bot.services." + service_name
            )
            service_class: Type[GenericService] = service_module.service_class
            self._services_classes[service_name] = service_class

    @timer(Timers.PERF_COUNTER)
    def _create_runners(self) -> None:
        """Creates runner for each available GPU."""
        for device in self._allocated_gpus:
            for runner_id in range(1):
                runner: Runner = Runner(
                    stop_run=self._stop_run,
                    mode=self._mode,
                    device=device,
                    runner_id=runner_id,
                    runner_status_queue=self._runner_status_queue,
                    runner_ect_queue=self._runner_ect_queue,
                    services_classes=self._services_classes,
                    results_queue=self._results_queue,
                    clients=self._clients,
                )
                runner.daemon = True
                runner.start()

                self._runners[cantor_pairing(device, runner_id)] = runner
                logger.debug(
                    f"{self._name}: Created runner {runner_id} in GPU {device}"
                )

    @timer(Timers.PERF_COUNTER)
    def _allocate_task(self) -> None:
        """Allocates a task to a runner based on the task scheduler."""
        task, runner_id = self._task_scheduler.schedule()
        logger.debug(pformat(task))

        runner: Runner = self._runners[runner_id]
        runner.task_in.send(task)

    @timer(Timers.PERF_COUNTER)
    def _check_result(
        self,
        results_queue: Queue[TaskMessage],
        req_stream: ManagerRequestsStream,
        req_pubsub: ManagerRequestsPubSub,
    ) -> None:
        """Checks the results of completed tasks and sends them to the clients.

        Args:
            results_queue (`Queue`[TaskMessage]): Queue where incoming task
                results are stored.
            req_stream (`ManagerRequestsStream`): Redis stream thread that
                1. Receives task requests and stores them in the requests queue.
                2. Sends task results to the appropriate client.
            req_pubsub (`ManagerRequestsPubSub`): Redis pubsub thread that
                1. Receives task requests and stores them in the requests queue.
                2. Sends task results to the appropriate client.
        """
        tasks_complete = 0
        tasks_failed = 0
        while True:
            result: TaskMessage = results_queue.get()

            if result["status"] == State.SUCCESS.value:
                tasks_complete += 1
            else:
                tasks_failed += 1

            logger.info(  # type: ignore
                f"{self._name}: {tasks_complete} task(s) complete!"
            )
            if self._tasks_failed > 0:
                logger.error(f"{self._name}: {tasks_failed} task(s) failed!")

            msg: Dict[str, str] = {"message": json.dumps(result)}
            req_stream.publish(result["channel"], msg)
            req_pubsub.publish(result["channel"], json.dumps(result))
