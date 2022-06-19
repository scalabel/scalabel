from enum import Enum


class ModelStatus(Enum):
    INVALID = 0
    LOADING = 1
    READY = 2
    IDLE = 3