# -*- coding: utf-8 -*-
import json
from time import sleep
from typing import Dict, List
from multiprocessing import Event, Process
from threading import Thread

from scalabel_bot.common.consts import (
    ConnectionRequest,
    CONNECTION_TIMEOUT,
    REDIS_HOST,
    REDIS_PORT,
    ResponseStatus,
    Timers,
)
from scalabel.common.logger import logger
from scalabel_bot.common.message import ConnectionMessage, Message
from scalabel_bot.profiling.timer import timer
from scalabel_bot.server.pubsub import ManagerConnectionsPubSub
from scalabel_bot.server.stream import ManagerConnectionsStream


class ClientManager(Process):
    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        stop_run: Event,
        clients: Dict[str, int],
    ) -> None:
        super().__init__()
        self._name = self.__class__.__name__
        self._stop_run: Event = stop_run
        self._clients: Dict[str, int] = clients
        self._clients_queue: List[Message] = []

    def run(self) -> None:
        self._conn_stream: ManagerConnectionsStream = ManagerConnectionsStream(
            stop_run=self._stop_run,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._clients_queue,
        )
        self._conn_stream.daemon = True
        self._conn_stream.start()
        self._conn_pubsub: ManagerConnectionsPubSub = ManagerConnectionsPubSub(
            stop_run=self._stop_run,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._clients_queue,
        )
        self._conn_pubsub.daemon = True
        self._conn_pubsub.start()
        # self._clients_checker = Thread(target=self._check_clients)
        # self._clients_checker.daemon = True
        # self._clients_checker.start()
        try:
            while not self._stop_run.is_set():
                if self._clients_queue:
                    self._connect(self._clients_queue.pop(0))
        except KeyboardInterrupt:
            self._conn_stream.shutdown()

    @timer(Timers.PERF_COUNTER)
    def _connect(self, conn: ConnectionMessage) -> None:
        if conn["request"] == str(ConnectionRequest.CONNECT):
            logger.info(
                "ClientManager: Received connection request from"
                f" client {conn['clientId']}"
            )
            self._clients[conn["clientId"]] = CONNECTION_TIMEOUT
        elif conn["request"] == str(ConnectionRequest.PING):
            logger.debug(
                f"ClientManager: Received ping from client {conn['clientId']}"
            )
            self._clients[conn["clientId"]] = CONNECTION_TIMEOUT
        elif conn["request"] == str(ConnectionRequest.DISCONNECT):
            logger.debug(
                "ClientManager: Received disconnection request from"
                f" client {conn['clientId']}"
            )
            if conn["clientId"] in self._clients:
                del self._clients[conn["clientId"]]
        else:
            logger.debug(
                "ClientManager: Unknown connection request from"
                f" client {conn['clientId']}"
            )
            return
        resp: ConnectionMessage = {
            "clientId": conn["clientId"],
            "status": str(ResponseStatus.OK),
            "requestsChannel": "REQUESTS",
            "host": REDIS_HOST,
            "port": REDIS_PORT,
        }
        msg: Dict[str, str] = {"message": json.dumps(resp)}
        self._conn_stream.publish(conn["channel"], msg)
        self._conn_pubsub.publish(conn["channel"], json.dumps(resp))

    def _check_clients(self) -> None:
        try:
            while not self._stop_run.is_set():
                sleep(1)
                if self._clients:
                    for client_id in self._clients.keys():
                        if self._clients[client_id] == 0:
                            logger.info(
                                f"Lost connection with client {client_id}"
                            )
                            del self._clients[client_id]
                        else:
                            self._clients[client_id] -= 1
        except KeyboardInterrupt:
            return

    def ready(self) -> bool:
        return self._conn_stream.ready
