# -*- coding: utf-8 -*-
"""PipeSwitch Redis Servers

This module implements classes for Redis servers to be used
by the manager, runners, and clients.

Todo:
    * None
"""

from threading import Event, Thread
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from typing import List, Dict, Tuple
from redis import Redis
from pprint import pformat, pprint
import json

from scalabel_bot.common.consts import Timers

from scalabel_bot.common.logger import logger
from scalabel_bot.profiling.timer import timer


class ManagerRequestsStream:
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
        host: str,
        port: int,
        sub_queue: "Queue[Dict[str, object]]" = Queue(),
        client_id: str = "",
    ) -> None:
        super().__init__()
        self._ready: bool = False
        self._host: str = host
        self._port: int = port
        self._client_id: str = client_id
        self._sub_queue: "Queue[Dict[str, object]]" = sub_queue
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
        listen = Thread(target=self._listen)
        listen.daemon = True
        listen.start()

    def _listen(self):
        while True:
            msg: Tuple[str, Dict[str, object]] = self._redis.xread(
                streams={
                    self.sub_stream: self._msg_id if self._msg_id else "$"
                },
                count=None,
                block=0,
            )
            if len(msg) > 0:
                self._process_msg(msg)

    @timer(Timers.PERF_COUNTER)
    def publish(self, stream: str, msg: Dict[str, object]) -> None:
        logger.spam(
            f"{self._server_name}: Publishing msg to stream"
            f" {stream}\n{pformat(msg)}"
        )
        self._redis.xadd(stream, msg)

    @timer(Timers.PERF_COUNTER)
    def delete_streams(self) -> None:
        logger.debug(
            f"{self._server_name}: Deleting stream: {self.sub_stream}"
        )
        self._redis.delete(self.sub_stream)

    @timer(Timers.PERF_COUNTER)
    def _process_msg(self, msg: List[object]) -> None:
        logger.spam(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            entry_id, entry = msg_item[1][0]
            if "message" in entry.keys():
                data = json.loads(entry["message"])
                self._sub_queue.put(data)
            else:
                self._sub_queue.put(msg[0][1][0][1])
            self._msg_id = entry_id
        logger.spam(f"{self._server_name}: Deleting message\n{pformat(msg)}")
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    def shutdown(self) -> None:
        self.delete_streams()

    @property
    def _server_name(self) -> str:
        if self._client_id != "":
            return f"{self.__class__.__name__}-{self._client_id}"
        else:
            return self.__class__.__name__

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    @abstractmethod
    def sub_stream(self) -> str:
        return "REQUESTS"


class ManagerRequestsPubSub:
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
        host: str,
        port: int,
        sub_queue: "Queue[Dict[str, object]]" = Queue(),
        client_id: str = "",
    ) -> None:
        super().__init__()
        self._ready: bool = False
        self._host: str = host
        self._port: int = port
        self._client_id: str = client_id
        self._sub_queue: "Queue[Dict[str, object]]" = sub_queue
        self._msg_id: str = ""
        self._redis = Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
            retry_on_timeout=True,
        )
        self._pubsub = self._redis.pubsub()

    def run(self) -> None:
        if self._redis.ping():
            logger.info(
                f"{self._server_name}: Listening to stream {self.sub_stream}"
            )
            self._ready = True
        else:
            logger.error(f"{self._server_name}: connection failed!")
        listen = Thread(target=self._listen)
        listen.daemon = True
        listen.start()

    def _listen(self):
        self._pubsub.subscribe(self.sub_stream)
        while True:
            for msg in self._pubsub.listen():
                if msg is not None:
                    if msg["data"] == 1:
                        logger.info(
                            f"{self._server_name}: subscribed to channel"
                            f" {msg['channel']}"
                        )
                    else:
                        logger.debug(
                            f"{self._server_name}: msg received from channel {msg['channel']}"
                        )
                        self._sub_queue.put(json.loads(msg["data"]))

    @timer(Timers.PERF_COUNTER)
    def publish(self, channel: str, msg: Dict[str, object]) -> None:
        self._redis.publish(channel, msg)

    @property
    def _server_name(self) -> str:
        if self._client_id != "":
            return f"{self.__class__.__name__}-{self._client_id}"
        return self.__class__.__name__

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    @abstractmethod
    def sub_stream(self) -> str:
        return "REQUESTS"


# class ManagerRequestsStream(RedisServer):
#     """Server in the manager that:
#     - Receives inference requests from a specific client.
#     - Sends inference results to the specific client.
#     """

#     @property
#     def sub_stream(self) -> str:
#         return "REQUESTS"


class ClientRequestsStream:
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
        host: str,
        port: int,
        sub_queue: "Queue[Dict[str, object]]" = Queue(),
        client_id: str = "",
    ) -> None:
        super().__init__()
        self._ready: bool = False
        self._host: str = host
        self._port: int = port
        self._client_id: str = client_id
        self._sub_queue: "Queue[Dict[str, object]]" = sub_queue
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
        listen = Thread(target=self._listen)
        listen.daemon = True
        listen.start()

    def _listen(self):
        while True:
            msg: Tuple[str, Dict[str, object]] = self._redis.xread(
                streams={
                    self.sub_stream: self._msg_id if self._msg_id else "$"
                },
                count=None,
                block=0,
            )
            if len(msg) > 0:
                self._process_msg(msg)

    @timer(Timers.PERF_COUNTER)
    def publish(self, stream: str, msg: Dict[str, object]) -> None:
        logger.spam(
            f"{self._server_name}: Publishing msg to stream"
            f" {stream}\n{pformat(msg)}"
        )
        self._redis.xadd(name=stream, fields=msg)

    @timer(Timers.PERF_COUNTER)
    def delete_streams(self) -> None:
        logger.debug(
            f"{self._server_name}: Deleting stream: {self.sub_stream}"
        )
        self._redis.delete(self.sub_stream)

    @timer(Timers.PERF_COUNTER)
    def _process_msg(self, msg: List[object]) -> None:
        logger.spam(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            entry_id, entry = msg_item[1][0]
            data = json.loads(entry["message"])
            self._sub_queue.put(data)
            self._msg_id = entry_id
        logger.spam(f"{self._server_name}: Deleting message\n{pformat(msg)}")
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    def shutdown(self) -> None:
        self.delete_streams()

    @property
    def _server_name(self) -> str:
        if self._client_id != "":
            return f"{self.__class__.__name__}-{self._client_id}"
        return self.__class__.__name__

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def pub_stream(self) -> str:
        return "REQUESTS"

    @property
    def sub_stream(self) -> str:
        return f"RESPONSES_{self._server_name}"


class ClientRequestsPubSub:
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
        host: str,
        port: int,
        sub_queue: "Queue[Dict[str, object]]" = Queue(),
        client_id: str = "",
    ) -> None:
        super().__init__()
        self._ready: bool = False
        self._host: str = host
        self._port: int = port
        self._client_id: str = client_id
        self._sub_queue: "Queue[Dict[str, object]]" = sub_queue
        self._msg_id: str = ""
        self._redis = Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
            retry_on_timeout=True,
        )
        self._pubsub = self._redis.pubsub()

    def run(self) -> None:
        if self._redis.ping():
            logger.info(
                f"{self._server_name}: Listening to stream {self.sub_stream}"
            )
            self._ready = True
        else:
            logger.error(f"{self._server_name}: connection failed!")
        listen = Thread(target=self._listen)
        listen.daemon = True
        listen.start()

    def _listen(self):
        self._pubsub.subscribe(self.sub_stream)
        while True:
            for msg in self._pubsub.listen():
                if msg is not None:
                    if msg["data"] == 1:
                        logger.info(
                            f"{self._server_name}: subscribed to channel"
                            f" {msg['channel']}"
                        )
                    else:
                        logger.debug(
                            f"{self._server_name}: msg received from channel {msg['channel']}"
                        )
                        self._sub_queue.put(json.loads(msg["data"]))

    @timer(Timers.PERF_COUNTER)
    def publish(self, channel: str, msg: Dict[str, object]) -> None:
        self._redis.publish(channel, msg)

    @property
    def _server_name(self) -> str:
        if self._client_id != "":
            return f"{self.__class__.__name__}-{self._client_id}"
        return self.__class__.__name__

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def pub_stream(self) -> str:
        return "REQUESTS"

    @property
    def sub_stream(self) -> str:
        return f"RESPONSES_{self._server_name}"


# class ManagerNotificationsPubSub:
#     """Server in the manager that:
#     - Receives inference requests from a specific client.
#     - Sends inference results to the specific client.
#     """

#     @timer(Timers.PERF_COUNTER)
#     def __init__(
#         self,
#         host: str,
#         port: int,
#         sub_queue: "Queue[Dict[str, object]]" = Queue(),
#         client_id: str = "",
#     ) -> None:
#         super().__init__()
#         self._ready: bool = False
#         self._stop_run: Event = Event()
#         self._host: str = host
#         self._port: int = port
#         self._client_id: str = client_id
#         self._sub_queue: "Queue[Dict[str, object]]" = sub_queue
#         self._redis = Redis(
#             host=self._host,
#             port=self._port,
#             encoding="utf-8",
#             decode_responses=True,
#             retry_on_timeout=True,
#         )
#         self._msg_id: str = ""

#     def publish(self, channel: str, msg: str) -> None:
#         self._redis.publish(channel, msg)

#     @property
#     def _server_name(self) -> str:
#         if self._client_id != "":
#             return f"{self.__class__.__name__}-{self._client_id}"
#         else:
#             return self.__class__.__name__


# class ManagerRequestsPubSub(Thread):
#     """Server in the manager that:
#     - Receives inference requests from a specific client.
#     - Sends inference results to the specific client.
#     """

#     @timer(Timers.PERF_COUNTER)
#     def __init__(
#         self,
#         host: str,
#         port: int,
#         sub_queue: "Queue[Dict[str, object]]" = Queue(),
#         client_id: str = "",
#     ) -> None:
#         super().__init__()
#         self._ready: bool = False
#         self._stop_run: Event = Event()
#         self._host: str = host
#         self._port: int = port
#         self._client_id: str = client_id
#         self._sub_queue: "Queue[Dict[str, object]]" = sub_queue
#         self._redis = Redis(
#             host=self._host,
#             port=self._port,
#             encoding="utf-8",
#             decode_responses=True,
#             retry_on_timeout=True,
#         )
#         self._pubsub = self._redis.pubsub()
#         self._msg_id: str = ""

#     def run(self) -> None:
#         self._pubsub.subscribe(self.sub_stream)
#         for msg in self._pubsub.listen():
#             if msg is not None:
#                 if msg["data"] == 1:
#                     logger.info(
#                         f"{self._server_name}: subscribed to channel"
#                         f" {msg['channel']}"
#                     )
#                 else:
#                     logger.debug(
#                         f"{self._server_name}: msg received from channel {msg['channel']}"
#                     )
#                     self._sub_queue.put(msg["data"])

#     def publish(self, channel: str, msg: str) -> None:
#         self._redis.publish(channel, msg)

#     @property
#     def _server_name(self) -> str:
#         if self._client_id != "":
#             return f"{self.__class__.__name__}-{self._client_id}"
#         else:
#             return self.__class__.__name__

#     @property
#     def sub_stream(self) -> str:
#         return "REQUESTS"
