# -*- coding: utf-8 -*-
"""PipeSwitch Manager.

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
    List,
    Dict,
    Tuple,
)
from multiprocessing.managers import DictProxy
from torch.multiprocessing import Manager, Process, Queue
from threading import Thread
from pprint import pformat
import json

from scalabel_bot.common.consts import (
    MODEL_LIST_FILE,
    REDIS_HOST,
    REDIS_PORT,
    State,
    Timers,
)
from scalabel_bot.common.exceptions import GPUError

from scalabel_bot.common.logger import logger
from scalabel_bot.common.servers import (
    ManagerRequestsPubSub,
    ManagerRequestsStream,
)

# from scalabel_bot.manager.client_manager import (
#     ClientManager,
# )
from scalabel_bot.manager.gpu_resource_allocator import (
    GPUResourceAllocator,
)
from scalabel_bot.manager.runner_scheduler import RunnerScheduler
from scalabel_bot.manager.task_scheduler import TaskScheduler
from scalabel_bot.profiling.timer import timer
from scalabel_bot.runner.runner import Runner


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
    def __init__(
        self, mode: str = "gpu", num_gpus: int = -1, gpu_ids: List[int] = []
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._do_run: bool = True
        self._mode: str = mode
        self._num_gpus: int = num_gpus
        self._gpu_ids: List[int] = gpu_ids
        self._manager = Manager()
        self._runner_status: DictProxy = self._manager.dict()
        self._runner_status_queue: "Queue[Tuple[int, State]]" = Queue()
        self._requests_queue: "Queue[Dict[str, object]]" = Queue()
        self._req_stream: ManagerRequestsStream = ManagerRequestsStream(
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._requests_queue,
        )
        self._req_pubsub: ManagerRequestsPubSub = ManagerRequestsPubSub(
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._requests_queue,
        )
        self._results_queue: "Queue[Dict[str, object]]" = Queue()
        self._model_classes: DictProxy = self._manager.dict()
        self._models: Dict[str, object] = {}
        self._runners: Dict[int, Runner] = {}
        self._tasks_complete: int = 0
        self._tasks_failed: int = 0

        self._req_stream.run()
        self._req_pubsub.run()
        if self._mode == "gpu":
            self._gra: GPUResourceAllocator = GPUResourceAllocator()
            self._allocated_gpus = self._gra.reserve_gpus(
                self._num_gpus, self._gpu_ids
            )
            logger.info(f"{self._name}: Allocated GPUs {self._allocated_gpus}")
            self._gra.warmup_gpus(gpus=self._allocated_gpus)
        else:
            self._allocated_gpus = [*range(self._num_gpus)]
        self._load_models()
        self._runner_scheduler: RunnerScheduler = RunnerScheduler(
            num_runners=self._allocated_gpus,
            runner_status=self._runner_status,
            runner_status_queue=self._runner_status_queue,
        )
        self._runner_scheduler.daemon = True
        self._runner_scheduler.start()
        self._create_runners()
        self._task_scheduler: TaskScheduler = TaskScheduler()
        check_result = Thread(target=self._check_result)
        check_result.daemon = True
        check_result.start()

    def run(self) -> None:
        """Main manager function that sets up the manager and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        try:
            # while not self._req_stream.ready:
            #     sleep(0.001)
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
                self._requests_queue = self._task_scheduler.reschedule(
                    self._requests_queue
                )
                task: Dict[str, object] = self._requests_queue.get()
                logger.debug(pformat(task))
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
        if hasattr(self, "_scheduler"):
            if self._runner_scheduler.is_alive():
                self._runner_scheduler.shutdown()
                self._runner_scheduler.terminate()
            logger.warning(f"{self._name}: Terminated scheduler")
        if hasattr(self, "_runners"):
            for runner in self._runners.values():
                if runner.is_alive():
                    runner.shutdown()
                    runner.terminate()
            logger.warning(f"{self._name}: Terminated runners")
        self._do_run = False
        if self._mode == "gpu" and hasattr(self, "_gra"):
            self._gra.release_gpus()
        logger.info(f"{self._name}: Successfully shut down")

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
                "scalabel_bot.task." + model_name
            )
            model_class: object = model_module.MODEL_CLASS
            self._model_classes[model_name] = model_class

    @timer(Timers.PERF_COUNTER)
    def _create_runners(self) -> None:
        """Create runner for each available GPU."""
        for runner_id in self._allocated_gpus:
            runner: Runner = Runner(
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
    def _allocate_task(self, task: Dict[str, object]) -> None:
        runner_id = self._runner_scheduler.schedule()
        runner: Runner = self._runners[runner_id]
        # logger.debug(
        #     f"{self._name}: Assigning task"
        #     f" {task['model_name']} {task['task_type']} with id"
        #     f" {task['task_id']} from client {task['client_id']} to"
        #     f" runner {runner_id}"
        # )
        runner.task_in.send(task)

    @timer(Timers.PERF_COUNTER)
    def _check_result(self) -> None:
        while True:
            result: Dict[str, object] = self._results_queue.get()
            if result["status"] == State.SUCCESS:
                self._tasks_complete += 1
            else:
                self._tasks_failed += 1
            logger.success(
                f"{self._name}: {self._tasks_complete} task(s) complete!"
            )
            if self._tasks_failed > 0:
                logger.error(
                    f"{self._name}: {self._tasks_failed} task(s) failed!"
                )
            result["status"] = result["status"].value
            msg: Dict[str, object] = {"message": json.dumps(result)}
            self._req_stream.publish(result["channel"], msg)
            self._req_pubsub.publish(result["channel"], json.dumps(result))
