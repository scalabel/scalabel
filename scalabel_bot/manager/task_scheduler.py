from abc import ABC, abstractmethod
from torch.multiprocessing import Queue
from typing import Dict


class Policy(ABC):
    def __init__(self) -> None:
        pass

    # @abstractmethod
    # def select_next(self, runners_list: List[int]) -> int:
    #     pass


class TaskScheduler:
    def __init__(self):
        pass

    def reschedule(
        self,
        task_queue: "Queue[Dict[str, object]]",
    ) -> "Queue[Dict[str, object]]":
        # task_list = list(task_queue)
        # for task in task_list:
        #     pass

        # new_task_queue = Queue()
        # for task in task_list:
        #     new_task_queue.put(task)

        return task_queue
