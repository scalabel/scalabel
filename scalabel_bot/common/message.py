from typing import Any, List, Dict, TypedDict


class Message(TypedDict, total=False):
    """Generic message template.

    Args:
        total (bool, optional): Whether all fields are mandatory when
            instantiating the dictionary. Defaults to False.
    """

    pass


class ConnectionMessage(Message):
    """Generic message template to establish connections."""

    clientId: str
    channel: str
    request: str
    status: str
    requestsChannel: str
    host: str
    port: int
    errMsg: str


class TaskMessage(Message):
    """Generic message template for task requests and results."""

    clientId: str
    mode: str
    type: str
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
    status: str
