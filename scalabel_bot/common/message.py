from typing import Any, List, Dict, TypedDict


class Message(TypedDict):
    clientId: str


class ConnectionMessage(Message, total=False):
    channel: str
    request: str
    status: str
    requestsChannel: str
    host: str
    port: int


class TaskMessage(Message, total=False):
    mode: str
    type: str
    projectName: str
    taskId: str
    taskType: str
    taskKey: str
    dataSize: int
    items: List[Dict[str, Any]]
    modelName: str
    ect: int
    wait: int
    channel: str
    output: object | None
    status: str
