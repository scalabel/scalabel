# -*- coding: utf-8 -*-
"""Scalabel Bot Client Manager.

This module communicates with clients regarding connections.

Todo:
    * None
"""


import json
from time import sleep
from typing import Dict, List
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as EventClass

# from threading import Thread

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
    """Manager subprocess that manages the connections with clients.

    Attributes:
        _name (`str`): Name of the class.
        _stop_run (`EventClass`): Manager run flag.
        _clients (`Dict[str, int]`): Dictionary of client IDs and their TTLs.
        _clients_queue (`List[str]`): Queue of incoming client IDs.
        _conn_stream (`ManagerConnectionsStream`): Redis stream thread that
            1. Receives connection requests from clients.
            2. Sends connection handshakes to successfully connected clients.
        _conn_pubsub (`ManagerConnectionsPubSub`): Redis pubsub thread that
            1. Receives connection requests from clients.
            2. Sends connection handshakes to successfully connected clients.

    Returns:
        _type_: _description_
    """

    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        clients: Dict[str, int],
        requests_channel: str,
    ) -> None:
        """Initializes the ClientManager class.

        Args:
            stop_run (`EventClass`): Global run flag.
            clients (`Dict[str, int]`): Dictionary of client IDs and their TTLs.
        """
        super().__init__()
        self._name = self.__class__.__name__
        self._stop_run: EventClass = Event()
        self._clients: Dict[str, int] = clients
        self._clients_queue: List[Message] = []
        self._requests_channel: str = requests_channel

    def run(self) -> None:
        """Sets up the client manager and runs it."""
        try:
            self._conn_stream: ManagerConnectionsStream = (
                ManagerConnectionsStream(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    sub_queue=self._clients_queue,
                )
            )
            self._conn_stream.daemon = True
            self._conn_stream.start()
            self._conn_pubsub: ManagerConnectionsPubSub = (
                ManagerConnectionsPubSub(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    sub_queue=self._clients_queue,
                )
            )
            self._conn_pubsub.daemon = True
            self._conn_pubsub.start()
            # self._clients_checker = Thread(target=self._check_clients)
            # self._clients_checker.daemon = True
            # self._clients_checker.start()

            while not self._stop_run.is_set():
                if self._clients_queue:
                    self._connect(self._clients_queue.pop(0))

        except KeyboardInterrupt:
            self._stop_run.set()
            self._conn_stream.shutdown()
            return

    @timer(Timers.PERF_COUNTER)
    def _connect(self, conn: ConnectionMessage) -> None:
        """Processes connection request of a new or existing client.

        Args:
            conn (`ConnectionMessage`): Type of connection request.
        """
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
            "requestsChannel": self._requests_channel,
            "host": REDIS_HOST,
            "port": REDIS_PORT,
        }
        msg: Dict[str, str] = {"message": json.dumps(resp)}
        self._conn_stream.publish(conn["handshakeChannel"], msg)
        self._conn_pubsub.publish(conn["handshakeChannel"], json.dumps(resp))

    def _check_clients(self) -> None:
        """Checks if existing clients are still connected."""
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
        """Checks if the client manager is ready to receive connections.

        Returns:
            bool: True if the client manager is ready to receive connections,
                False otherwise.
        """
        return self._conn_stream.ready
