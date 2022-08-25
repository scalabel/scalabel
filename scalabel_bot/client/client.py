# -*- coding: utf-8 -*-
import sys
import json
from time import perf_counter_ns, sleep
from argparse import ArgumentParser
from uuid import uuid4
from queue import Queue
from typing import Dict, List
from threading import Thread
from multiprocessing import Event
from multiprocessing.synchronize import Event as EventClass

from scalabel_bot.client.task_request import (
    get_image_item,
    get_opt_items,
)
from scalabel_bot.common.consts import (
    ConnectionRequest,
    ESTCT,
    MODELS,
    REDIS_HOST,
    REDIS_PORT,
    ResponseStatus,
    State,
    Timers,
)
from scalabel.common.logger import logger
from scalabel_bot.common.message import ConnectionMessage, TaskMessage
from scalabel_bot.profiling.timer import timer
from scalabel_bot.server.stream import (
    ClientConnectionsStream,
    ClientRequestsStream,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="FsDet demo for builtin models")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet152",
        help="Name of model for processing requests. Default is 'resnet152'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size of data. Default is 8.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=1,
        help="Number of iterations of requests. Default is 1.",
    )
    return parser


class Client:
    @timer(Timers.PERF_COUNTER)
    def __init__(self, model_name, batch_size, num_tasks) -> None:
        super().__init__()
        self._stop_run: EventClass = Event()
        self._name: str = self.__class__.__name__
        self._client_id: str = str(uuid4())
        self._results_queue: List[TaskMessage] = []
        self._handshakes_queue: List[ConnectionMessage] = []
        self._conn_server: ClientConnectionsStream = ClientConnectionsStream(
            idx=self._client_id,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._handshakes_queue,
        )
        self._requests_server: ClientRequestsStream = ClientRequestsStream(
            idx=self._client_id,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._results_queue,
        )
        self._requests_channel: str = ""
        self._model_name: str = model_name
        self._batch_size: int = batch_size
        self._num_tasks: int = num_tasks
        self._task_queue: Queue[TaskMessage] = Queue()
        self._pending_tasks: Dict[str, TaskMessage] = {}

    @timer(Timers.PERF_COUNTER)
    def run(self) -> None:
        try:
            self._conn_server.daemon = True
            self._conn_server.start()
            while not self._conn_server.ready:
                sleep(0.1)
            self._requests_server.daemon = True
            self._requests_server.start()
            self._connect(ConnectionRequest.CONNECT)
            self._prepare_requests()
            while not self._requests_server.ready:
                sleep(0.1)
            # ping = Thread(target=self._ping)
            # ping.daemon = True
            # ping.start()
            send_requests = Thread(target=self._send_requests)
            send_requests.daemon = True
            send_requests.start()
            self._receive_results()
            self._connect(ConnectionRequest.DISCONNECT)
        except KeyboardInterrupt:
            self._shutdown()

    @timer(Timers.PERF_COUNTER)
    def _connect(self, mode: ConnectionRequest) -> None:
        conn: ConnectionMessage = {
            "clientId": self._client_id,
            "handshakeChannel": self._conn_server.sub_stream,
            "request": str(mode),
        }
        msg: Dict[str, str] = {"message": json.dumps(conn)}
        channel: str = self._conn_server.pub_stream
        self._conn_server.publish(channel, msg)
        while not self._handshakes_queue:
            sleep(0.01)
        resp: ConnectionMessage = self._handshakes_queue.pop(0)
        if resp["clientId"] == self._client_id:
            if resp["status"] == str(ResponseStatus.OK):
                if mode == ConnectionRequest.CONNECT:
                    self._requests_channel = resp["requestsChannel"]
                    logger.info(  # type: ignore
                        f"Client {self._client_id}: Connected"
                    )
                elif mode == ConnectionRequest.PING:
                    logger.info(  # type: ignore
                        f"Client {self._client_id}: Pinged"
                    )
                elif mode == ConnectionRequest.DISCONNECT:
                    logger.info(  # type: ignore
                        f"Client {self._client_id}: Disconnected"
                    )
                return
            elif resp["status"] == str(ResponseStatus.ERROR):
                logger.error(
                    f"Client {self._client_id} error msg: {resp['errMsg']}"
                )
                sys.exit(1)
        else:
            logger.debug(
                f"Client {self._client_id}: Ignoring invalid handshake {msg}"
            )

    def _ping(self) -> None:
        while self._stop_run:
            sleep(5)
            self._connect(ConnectionRequest.PING)

    @timer(Timers.PERF_COUNTER)
    def _prepare_requests(self) -> None:
        for _ in range(self._num_tasks):
            task_id: str = str(uuid4())
            if self._model_name == "fsdet":
                task_type = "box2d"
            elif self._model_name == "dd3d":
                task_type = "box3d"
            elif self._model_name == "opt":
                task_type = "textgen"
            task_key: str = "projects/bot-batch/saved/000000"
            # task_key: str = "projects/image/saved/000025"

            if self._model_name == "opt":
                items = get_opt_items()
            else:
                items = get_image_item()

            mode = "inference"
            data_size = 1 if self._model_name == "opt" else 1
            task: TaskMessage = {
                "mode": mode,
                "clientId": self._client_id,
                "projectName": "bot-batch",
                "taskId": task_id,
                "taskType": task_type,
                "taskKey": task_key,
                "dataSize": data_size,
                "items": items,
                "modelName": self._model_name,
                "ect": ESTCT[mode][MODELS[task_type]] * data_size,
                "wait": 0,
                "channel": self._requests_server.sub_stream,
            }
            self._task_queue.put(task)

    def _send_requests(self) -> None:
        self._start_time = perf_counter_ns()
        while True:
            task: TaskMessage = self._task_queue.get()
            self._send_request(task)
            if task["modelName"] == "opt":
                sleep(1)
            else:
                sleep(0.05)

    @timer(Timers.THREAD_TIMER)
    def _send_request(self, task) -> None:
        logger.debug(
            f"{self._name} {self._client_id}: sending task"
            f" {task['modelName']} {task['taskType']} with id"
            f" {task['taskId']}"
        )
        # self._pending_tasks[task["taskId"]] = task
        msg: Dict[str, str] = {"message": json.dumps(task)}
        self._requests_server.publish(self._requests_channel, msg)

    @timer(Timers.PERF_COUNTER)
    def _receive_results(self) -> None:
        task_count = 0
        while task_count != self._num_tasks:
            if self._results_queue:
                result: TaskMessage = self._results_queue.pop(0)
                # if result["taskId"] in self._pending_tasks:
                if result["status"] == State.SUCCESS.value:
                    logger.info(  # type: ignore
                        f"{self._name} {self._client_id}: received task"
                        f" {result['modelName']} {result['taskType']} with"
                        f" id {result['taskId']} result"
                    )
                    logger.debug(result["output"])
                    task_count += 1
                    logger.info(
                        f"{self._name} {self._client_id}: completed task(s)"
                        f" {task_count}/{self._num_tasks}"
                    )
                    # self._pending_tasks.pop(result["taskId"])
                else:
                    # TODO: handle failed task
                    # TODO: push failed task back to the task queue and resend
                    logger.error(
                        f"{self._name} {self._client_id}: task"
                        f" {result['modelName']} {result['taskType']} with"
                        f" id {result['taskId']} failed"
                    )
                    logger.debug(
                        f"{self._name} {self._client_id}: retrying task"
                        f" {result['modelName']} {result['taskType']} with"
                        f" id {result['taskId']}"
                    )
                    self._task_queue.put(self._pending_tasks[result["taskId"]])
        total_time = perf_counter_ns() - self._start_time
        logger.info(f"Total time taken: {total_time / 1000000} ms")
        logger.info(
            f"Average task time: {total_time / self._num_tasks / 1000000} ms"
        )

    def _shutdown(self) -> None:
        logger.warning(f"{self._name} {self._client_id}: shutting down")


def launch():
    args: ArgumentParser = get_parser().parse_args()
    client = Client(
        model_name=args.model, batch_size=args.batch_size, num_tasks=args.tasks
    )
    client.run()


if __name__ == "__main__":
    launch()
