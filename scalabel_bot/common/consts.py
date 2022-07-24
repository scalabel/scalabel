from enum import Enum
from typing import Dict

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


class State(Enum):
    """Common status codes."""

    def __str__(self) -> str:
        return str(self.value)

    STARTUP = "STARTUP"
    IDLE = "IDLE"
    RESERVED = "RESERVED"
    BUSY = "BUSY"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"


class ResponseStatus(Enum):
    def __str__(self) -> str:
        return str(self.value)

    OK = "OK"
    WAIT = "WAIT"
    ERROR = "ERROR"


class ConnectionRequest(Enum):
    def __str__(self) -> str:
        return str(self.value)

    CONNECT = "CONNECT"
    DISCONNECT = "DISCONNECT"


class Timers(Enum):
    def __str__(self) -> str:
        return str(self.value)

    PERF_COUNTER = "PERF_COUNTER"
    PROCESS_TIMER = "PROCESS_TIMER"
    THREAD_TIMER = "THREAD_TIMER"


MODELS = {"box2d": "fsdet", "box3d": "dd3d", "textgen": "opt"}

ESTCT: Dict[str, Dict[str, int]] = {
    "inference": {"dd3d": 113, "fsdet": 172, "opt": 19980},
    "training": {},
}
