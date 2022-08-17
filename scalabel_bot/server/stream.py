from multiprocessing import Event
from threading import Thread
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from redis import Redis
from pprint import pformat
import json

from scalabel_bot.common.consts import ESTCT, MODELS, Timers
from scalabel.common.logger import logger
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.profiling.timer import timer


class Stream(Thread, ABC):
    """Redis Server abstract base class.

    It has two main functions:
        - Subscribes to one channel and receives messages.
        - Publishes to another channel.

    Attributes:
        _server_name (`str`): Name of the server.
        _host (`str`): Redis host IP address.
        _port (`int`): Redis port number.
        _redis_pub (`redis.Redis`): Redis client for publishing.
        pub_stream (`str`): Redis channel to publish to.
        _pub_queue (`Queue[str]`): Queue for publishing messages.
        _redis_sub (`redis.Redis`): Redis client for subscribing.
        sub_stream (`str`): Redis channel to subscribe to.
        _sub_queue (`Queue[str]`): Queue for receiving messages.
        _module_id (`int`, optional): ID of the module the server is under.
        _worker_id (`int`, optional): ID of the worker the server is under.
        _client_id (`str`, optional): ID of the client.
    """

    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        stop_run: Event,
        host: str,
        port: int,
        sub_queue: List[TaskMessage],
        idx: str = "",
    ) -> None:
        super().__init__()
        self._stop_run: Event = stop_run
        self._ready: bool = False
        self._host: str = host
        self._port: int = port
        self._idx: str = idx
        self._sub_queue: List[TaskMessage] = sub_queue
        self._msg_id: str = ""
        self._redis = Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
            retry_on_timeout=True,
        )

    def run(self) -> None:
        if self._redis.ping():
            logger.info(
                f"{self._server_name}: Listening to stream {self.sub_stream}"
            )
            self._ready = True
        else:
            logger.error(f"{self._server_name}: connection failed!")
        self._listen()

    @timer(Timers.PERF_COUNTER)
    def publish(self, stream: str, msg: str) -> None:
        self._redis.xadd(stream, msg)

    def _listen(self) -> None:
        try:
            while not self._stop_run.is_set():
                msg: List[
                    List[str | List[Tuple[str, Dict[str, str]]]]
                ] = self._redis.xread(
                    streams={
                        self.sub_stream: self._msg_id if self._msg_id else "$"
                    },
                    count=None,
                    block=0,
                )
                if len(msg) > 0:
                    self._process_msg(msg)
        except KeyboardInterrupt:
            return

    @timer(Timers.PERF_COUNTER)
    def _delete_streams(self) -> None:
        logger.debug(
            f"{self._server_name}: Deleting stream: {self.sub_stream}"
        )
        self._redis.delete(self.sub_stream)

    @abstractmethod
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        pass

    def shutdown(self) -> None:
        self._delete_streams()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    @abstractmethod
    def pub_stream(self) -> str:
        pass

    @property
    @abstractmethod
    def sub_stream(self) -> str:
        pass

    @property
    def _server_name(self) -> str:
        if self._idx != "":
            return f"{self.__class__.__name__}-{self._idx}"
        return self.__class__.__name__


class ManagerConnectionsStream(Stream):
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        logger.debug(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            if isinstance(msg_item[1], str):
                continue
            entry_id, entry = msg_item[1][0]
            if "message" in entry.keys():
                data: TaskMessage = json.loads(entry["message"])
                self._sub_queue.append(data)
            self._msg_id = entry_id
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    @property
    def pub_stream(self) -> str:
        return "modelNotify"

    @property
    def sub_stream(self) -> str:
        return "modelRegister"


class ManagerRequestsStream(Stream):
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        logger.debug(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            if isinstance(msg_item[1], str):
                continue
            entry_id, entry = msg_item[1][0]
            if "message" in entry.keys():
                data: TaskMessage = json.loads(entry["message"])
                if "ect" not in data:
                    data["ect"] = (
                        ESTCT[data["mode"]][MODELS[data["taskType"]]]
                        * data["dataSize"]
                    )
                if "wait" not in data:
                    data["wait"] = 0
                self._sub_queue.append(data)
            self._msg_id = entry_id
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    @property
    def pub_stream(self) -> str:
        return super().pub_stream

    @property
    def sub_stream(self) -> str:
        return "requests"


class ClientConnectionsStream(Stream):
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        logger.debug(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            if isinstance(msg_item[1], str):
                continue
            entry_id, entry = msg_item[1][0]
            if "message" in entry.keys():
                data: TaskMessage = json.loads(entry["message"])
                self._sub_queue.append(data)
            self._msg_id = entry_id
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    @property
    def pub_stream(self) -> str:
        return "modelRegister"

    @property
    def sub_stream(self) -> str:
        return f"modelNotify_{self._server_name}"


class ClientRequestsStream(Stream):
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        logger.debug(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            if isinstance(msg_item[1], str):
                continue
            entry_id, entry = msg_item[1][0]
            if "message" in entry.keys():
                data: TaskMessage = json.loads(entry["message"])
                self._sub_queue.append(data)
            self._msg_id = entry_id
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    @property
    def pub_stream(self) -> str:
        return "requests"

    @property
    def sub_stream(self) -> str:
        return f"responses_{self._server_name}"
