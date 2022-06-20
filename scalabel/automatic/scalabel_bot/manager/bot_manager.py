# -*- coding: utf-8 -*-
"""PipeSwitch Manager

This module is the main entry point for the PipeSwitch Manager.

Example:
    Run this file from the main directory of the project::

        $ python3 manager/manager.py

Todo:
    * None
"""


from time import sleep
import os
import importlib
from typing import (  # pylint: disable=unused-import
    Any,
    List,
    OrderedDict,
    Tuple,
)
from multiprocessing.managers import DictProxy
from torch.multiprocessing import Manager, Process, Queue
from pprint import pprint
import json

from scalabel.automatic.scalabel_bot.common.consts import (
    MODEL_LIST_FILE,
    State,
    Timers,
)
from scalabel.automatic.scalabel_bot.common.exceptions import GPUError
from scalabel.automatic.scalabel_bot.common.logger import logger
from scalabel.automatic.scalabel_bot.manager.client_manager import (
    ClientManager,
)
from scalabel.automatic.scalabel_bot.manager.gpu_resource_allocator import (
    GPUResourceAllocator,
)
from scalabel.automatic.scalabel_bot.manager.scheduler import Scheduler
from scalabel.automatic.scalabel_bot.profiling.timer import timer
from scalabel.automatic.scalabel_bot.runner.runner import Runner


class BotManager:
    """Manager thread that acts as a middleman between clients and runners.

    It has two main functions:
        1. It receives requests from clients and sends them
           to the appropriate runner.
        2. It receives results from runners and sends them
           to the appropriate client.

    Attributes:
        do_run (`bool`): Manager run flag.
        gra (`GPUResourceAllocator`):
            Thread that polls for available GPUs and reserves them.
        req_server (`ManagerRequestsServer`):
            Redis server for receiving requests from clients
            and sending tasks to runners.
        res_server (ManagerResultsServer):
            Redis server for receiving results from runners
            and sending them to clients.
        model_list (`List[str]`): List of ML models to be loaded into memory.
        runners (`OrderedDict[int, RunnerThd]`): Database of runners.
        scheduler (`Scheduler`): Thread that schedules tasks to runners.
    """

    @timer(Timers.PERF_COUNTER)
    def __init__(self, mode: str = "gpu", num_gpus: int = -1) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._do_run: bool = True
        self._mode: str = mode
        self._num_gpus: int = num_gpus
        if self._mode == "gpu":
            self._gra: GPUResourceAllocator = GPUResourceAllocator()
        self._manager = Manager()
        self._runner_status: DictProxy = self._manager.dict()
        self._runner_status_queue: "Queue[Tuple[int, State]]" = Queue()
        self._requests_queue: "Queue[OrderedDict[str, object]]" = Queue()
        self._results_queue: "Queue[OrderedDict[str, object]]" = Queue()
        self._client_manager: ClientManager = ClientManager(
            requests_queue=self._requests_queue,
            results_queue=self._results_queue,
        )
        self._model_classes: DictProxy = self._manager.dict()
        self._models: OrderedDict[str, object] = OrderedDict()
        self._runners: OrderedDict[int, Runner] = OrderedDict()

    def run(self) -> None:
        """Main manager function that sets up the manager and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        try:
            self._setup()
            logger.debug(f"{self._name} setup done")
            while not self._client_manager.ready:
                sleep(0.001)
            logger.success(
                "\n*********************************************\n"
                f"{self._name}: Ready to receive requests!\n"
                "*********************************************"
            )
            logger.debug(
                f"{self._name}: Waiting for at least one runner to be ready"
            )
            while len(self._runner_status) < 1:
                sleep(1)
            logger.success(
                "\n******************************************\n"
                f"{self._name}: Ready to execute tasks!\n"
                "******************************************"
            )
            while self._do_run:
                task: OrderedDict[str, object] = self._requests_queue.get()
                # logger.info(
                #     f"{self._name}: {self._requests_queue.qsize() + 1} task(s)"
                #     " in requests queue"
                # )
                # logger.info(
                #     f"{self._name}: Received task"
                #     f" {task['model_name']} {task['task_type']} with id"
                #     f" {task['task_id']} from client {task['client_id']}"
                # )
                self._allocate_task(task)
        except GPUError:
            return
        except KeyboardInterrupt:
            return

    @timer(Timers.PERF_COUNTER)
    def shutdown(self) -> None:
        logger.warning(f"{self._name}: Shutting down")
        if hasattr(self, "_client_manager"):
            if self._client_manager.is_alive():
                self._client_manager.shutdown()
                self._client_manager.terminate()
            logger.warning(f"{self._name}: Terminated client manager")
        if hasattr(self, "_scheduler"):
            if self._scheduler.is_alive():
                self._scheduler.shutdown()
                self._scheduler.terminate()
            logger.warning(f"{self._name}: Terminated scheduler")
        if hasattr(self, "_runners"):
            for runner in self._runners.values():
                if runner.is_alive():
                    runner.shutdown()
                    runner.terminate()
            logger.warning(f"{self._name}: Terminated runners")
        self.do_run = False
        if self._mode == "gpu" and hasattr(self, "_gra"):
            self._gra.release_gpus()
        logger.info(f"{self._name}: Successfully shut down")

    @timer(Timers.PERF_COUNTER)
    def _setup(self) -> None:
        """Run setup tasks in the manager."""
        logger.debug(f"{self._name}: Start")
        self._client_manager.daemon = True
        self._client_manager.start()
        if self._mode == "gpu":
            self.allocated_gpus = self._gra.reserve_gpus(self._num_gpus)
            logger.info(f"{self._name}: Allocated GPUs {self.allocated_gpus}")
            self._gra.warmup_gpus(gpus=self.allocated_gpus)
        else:
            self.allocated_gpus = [*range(self._num_gpus)]
        self._load_models()
        self._scheduler: Scheduler = Scheduler(
            num_runners=self.allocated_gpus,
            runner_status=self._runner_status,
            runner_status_queue=self._runner_status_queue,
        )
        self._scheduler.daemon = True
        self._scheduler.start()
        self._create_runners()

    @timer(Timers.PERF_COUNTER)
    def _load_model_list(self, file_name: str) -> None:
        """Load a list of models to be used by the manager.

        Args:
            file_name (`str`): Path to the file containing the list of models.

        Raises:
            `AssertionError`: If the file does not exist.
        """
        if not os.path.exists(file_name):
            logger.error(
                f"{self._name}: Model list file {file_name} not found"
            )
            raise KeyboardInterrupt

        with open(file=file_name, mode="r", encoding="utf-8") as f:
            self._model_list: List[str] = [
                line.strip() for line in f.readlines()
            ]

    @timer(Timers.PERF_COUNTER)
    def _load_models(self) -> None:
        self._load_model_list(file_name=MODEL_LIST_FILE)
        for model_name in self._model_list:
            model_module = importlib.import_module(
                "pipeswitch.task." + model_name
            )
            model_class: object = model_module.MODEL_CLASS
            self._model_classes[model_name] = model_class

    @timer(Timers.PERF_COUNTER)
    def _create_runners(self) -> None:
        """Create runner for each available GPU.

        Args:
            num_workers (`int`, optional):
                number of workers to create per runner, default 2.
        """
        for runner_id in self.allocated_gpus:
            runner: Process = Runner(
                mode=self._mode,
                device=runner_id,
                runner_status_queue=self._runner_status_queue,
                model_list=self._model_list,
                model_classes=self._model_classes,
                results_queue=self._results_queue,
            )
            runner.daemon = True
            runner.start()
            self._runners[runner_id] = runner
            logger.debug(f"{self._name}: Created runner in GPU {runner_id}")

    @timer(Timers.PERF_COUNTER)
    def _allocate_task(self, task: OrderedDict[str, Any]) -> None:
        runner_id = self._scheduler.schedule()
        runner = self._runners[runner_id]
        # logger.debug(
        #     f"{self._name}: Assigning task"
        #     f" {task['model_name']} {task['task_type']} with id"
        #     f" {task['task_id']} from client {task['client_id']} to"
        #     f" runner {runner_id}"
        # )
        runner.task_in.send(task)
