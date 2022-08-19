from enum import Enum
from typing import Dict, TypeVar

T = TypeVar("T")

# Redis host address
REDIS_HOST = "127.0.0.1"
# Redis port number
REDIS_PORT = 6379

# model list file name
MODEL_LIST_FILE = "scalabel_bot/model_list.txt"
# debug log file name
DEBUG_LOG_FILE = "logs/debug/debug.log"
# timing log file name
TIMING_LOG_FILE = "logs/profiling/timing.log"
# latency threshold (in ms)
LATENCY_THRESHOLD = 10

CONNECTION_TIMEOUT = 10


class State(Enum):
    """Common status codes."""

    def __str__(self) -> str:
        return str(self.value)

    STARTUP = "startup"
    IDLE = "idle"
    RESERVED = "reserved"
    BUSY = "busy"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    TERMINATED = "terminated"


class ResponseStatus(Enum):
    def __str__(self) -> str:
        return str(self.value)

    OK = "ok"
    WAIT = "wait"
    ERROR = "error"


class ConnectionRequest(Enum):
    def __str__(self) -> str:
        return str(self.value)

    CONNECT = "connect"
    PING = "ping"
    DISCONNECT = "disconnect"


class Timers(Enum):
    def __str__(self) -> str:
        return str(self.value)

    PERF_COUNTER = "perf_counter"
    PROCESS_TIMER = "process_timer"
    THREAD_TIMER = "thread_timer"


class ServiceMode(Enum):
    def __str__(self) -> str:
        return str(self.value)

    INFERENCE = "inference"
    TRAIN = "train"
    NONE = "none"


MODELS = {"box2d": "fsdet", "box3d": "dd3d", "textgen": "opt"}

ESTCT: Dict[str, Dict[str, int]] = {
    "inference": {"dd3d": 113, "fsdet": 172, "opt": 1988},
    "training": {},
}
