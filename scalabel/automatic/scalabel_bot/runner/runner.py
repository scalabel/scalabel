# -*- coding: utf-8 -*-
"""PipeSwitch Runner

This module spawns workers and allocates tasks to available workers.

Todo:
    * None
"""
from time import sleep
from typing import (  # pylint: disable=unused-import
    Any,
    List,
    OrderedDict,
    Tuple,
)
import torch
from torch.multiprocessing import (  # pylint: disable=unused-import
    Event,
    Pipe,
    Process,
    Queue,
)
import jsonpickle
from pprint import pformat
from threading import Thread

from detectron2.structures import Boxes, Instances

from scalabel.automatic.scalabel_bot.common.consts import MODELS, State, Timers
from scalabel.automatic.scalabel_bot.common.logger import logger
from scalabel.automatic.scalabel_bot.profiling.timer import timer
from scalabel.automatic.scalabel_bot.runner.runner_common import ModelSummary


class Runner(Process):
    """Runner thread that receives tasks from the manager
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
        runner_status_queue: "Queue[Tuple[int, State]]",
        model_list: List[str],
        model_classes: OrderedDict[str, object],
        results_queue: "Queue[OrderedDict[str, Any]]",
    ) -> None:
        super().__init__()
        self._name = self.__class__.__name__
        self._stop_run: Event = Event()
        self._mode: str = mode
        self._device: int = device
        self._status: State = State.STARTUP
        self._runner_status_queue: "Queue[Tuple[int, State]]" = (
            runner_status_queue
        )
        self._task_in, self._task_out = Pipe()
        self._results_queue: "Queue[OrderedDict[str, Any]]" = results_queue
        self._model_list = model_list
        self._model_classes: OrderedDict[str, object] = model_classes
        self._models: OrderedDict[str, Any] = OrderedDict()

    def run(self) -> None:
        """Main runner function that sets up the runner and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        logger.debug(f"{self._name} {self._device}: start")
        if self._mode == "gpu":
            logger.debug(f"{self._name} {self._device}: share GPU memory")
            self._load_jobs: List[Thread] = []
            for model_name, model_class in self._model_classes.items():
                model_summary: ModelSummary = ModelSummary(
                    mode=self._mode,
                    devices=self._device,
                    model_name=model_name,
                    model_class=model_class,
                )
                load_model = Thread(target=model_summary.load_model)
                load_model.daemon = True
                load_model.start()
                load_model.join()
                self._load_jobs.append(load_model)
                self._models[model_name] = model_summary
            logger.debug(f"{self._name} {self._device}: import models")
        self._update_status(State.IDLE)
        while not self._stop_run.is_set():
            task: OrderedDict[str, Any] = self._task_out.recv()
            self._manage_task(task)

    @timer(Timers.THREAD_TIMER)
    def _update_status(self, status: State) -> None:
        """Updates own runner status based on worker statuses"""
        try:

            self._status = status
            logger.debug(
                f"{self._name} {self._device}: Updating status to"
                f" {self._status}"
            )
            self._runner_status_queue.put((self._device, self._status))
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.THREAD_TIMER)
    def _manage_task(self, task: OrderedDict[str, Any]) -> None:
        try:
            # logger.debug(
            #     f"{self._name} {self._device}: received task"
            #     f" {task['model_name']} {task['task_type']} with id"
            #     f" {task['task_id']} from client {task['client_id']}"
            # )
            self._update_status(State.BUSY)
            if self._mode == "gpu":
                model_summary: ModelSummary = self._models[
                    f"{MODELS[task['taskType']]}_{task['type']}"
                ]
                output = self._execute_task(task, model_summary)
            else:
                logger.debug(
                    f"{self._name} {self._device}: CPU debug mode task"
                    " execution"
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
            # task: OrderedDict[str, Any] = {
            #     "client_id": task["client_id"],
            #     "task_type": task["task_type"],
            #     "task_id": task["task_id"],
            #     "model_name": task["model_name"],
            #     "status": State.SUCCESS,
            #     "output": jsonpickle.encode(output),
            # }
            task["status"] = State.SUCCESS
            task["output"] = output
            self._results_queue.put(task)
            # logger.debug(
            #     f"{self._name} {self._device}: task"
            #     f" {task['task_id']} {task['task_type']} with id"
            #     f" {task['task_id']} complete"
            # )
            self._update_status(State.IDLE)
        except RuntimeError as runtime_err:
            logger.error(runtime_err)
            logger.error(f"{self._name} {self._device}: task failed!")
            msg: OrderedDict[str, Any] = {
                "client_id": task["client_id"],
                "task_type": task["task_type"],
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "status": State.FAILED,
            }
            self._results_queue.put(msg)
            self._update_status(State.IDLE)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.THREAD_TIMER)
    def _execute_task(
        self, task: OrderedDict[str, Any], model_summary: ModelSummary
    ) -> Any:
        """Executes a task."""
        data = model_summary.load_data(task)
        # logger.debug(
        #     f"{self._name} {self._device}: retrieved data for task"
        #     f" {task['model_name']} {task['task_type']} with id"
        #     f" {task['task_id']} from client {task['client_id']}"
        # )
        # logger.spam(f"{self._name} {self._device} data: \n{pformat(data)}")
        output = model_summary.execute(task, data)
        # logger.spam(f"{self._name} {self._device} output: \n{pformat(output)}")
        return output

    @timer(Timers.THREAD_TIMER)
    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name} {self._device}: stopping...")
        self._stop_run.set()
        if hasattr(self, "_load_jobs"):
            for load_job in self._load_jobs:
                if load_job.is_alive():
                    load_job.terminate()
        logger.debug(f"{self._name} {self._device}: stopped!")

    @property
    def task_in(self):
        return self._task_in