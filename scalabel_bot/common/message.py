from typing import Any, List, Dict, TypedDict


class Message(TypedDict, total=False):
    """Generic message template.

    Args:
        total (bool, optional): Whether all fields are mandatory when
            instantiating the dictionary. Defaults to False.
    """

    clientId: str
    status: str


class ConnectionMessage(Message, total=False):
    """Generic message template to establish connections.

    Args:
        total (bool, optional): Whether all fields are mandatory when
            instantiating the dictionary. Defaults to False.
    """

    handshakeChannel: str
    request: str
    requestsChannel: str
    host: str
    port: int
    errMsg: str


class TaskMessage(Message, total=False):
    """Generic message template for task requests and results.

    Args:
        total (bool, optional): Whether all fields are mandatory when
            instantiating the dictionary. Defaults to False.
    """

    mode: str
    projectName: str
    taskId: str
    taskType: str
    taskKey: str
    dataSize: int
    items: List[Dict[str, Any]]
    modelName: str
    channel: str
    ect: int
    wait: float
    output: object | None
