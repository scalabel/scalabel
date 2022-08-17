import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List

from scalabel_bot.common.consts import Timers
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.profiling.timer import timer


class SchedulingPolicy(ABC):
    def __init__(self, num_runners: int):
        self._num_runners = num_runners
        self._runner_count = 0

    @abstractmethod
    def choose_task(self, task_list: List[TaskMessage]) -> TaskMessage:
        pass

    @abstractmethod
    def choose_resource(self, runner_ect: Dict[int, int]) -> int:
        pass


class RoundRobin(SchedulingPolicy):
    def choose_task(self, task_list: List[TaskMessage]) -> TaskMessage:
        return task_list.pop(0)

    @timer(Timers.THREAD_TIMER)
    def choose_resource(self, runner_ect: Dict[int, int]) -> int:
        self._runner_count = (self._runner_count + 1) % len(runner_ect)
        return list(runner_ect)[self._runner_count]


class ShortestJobFirst(SchedulingPolicy):
    def choose_task(self, task_list: List[TaskMessage]) -> TaskMessage:
        next_task = min(task_list, key=lambda x: x["ect"])
        task_list.remove(next_task)
        return next_task

    def choose_resource(self, runner_ect: Dict[int, int]) -> int:
        self._runner_count = (self._runner_count + 1) % len(runner_ect)
        return list(runner_ect)[self._runner_count]


class LoadBalancing(SchedulingPolicy):
    def choose_task(self, task_list: List[TaskMessage]) -> TaskMessage:
        return task_list.pop(0)

    @timer(Timers.THREAD_TIMER)
    def choose_resource(self, runner_ect: Dict[int, int]) -> int:
        next_runner_id = min(runner_ect, key=runner_ect.get)
        return next_runner_id


class SJFLB(SchedulingPolicy):
    def choose_task(self, task_list: List[TaskMessage]) -> TaskMessage:
        next_task = min(task_list, key=lambda x: x["ect"])
        task_list.remove(next_task)
        return next_task

    def choose_resource(self, runner_ect: Dict[int, int]) -> int:
        next_runner_id = min(runner_ect, key=runner_ect.get)
        return next_runner_id


class SJFLBAP(SchedulingPolicy):
    def choose_task(self, task_list: List[TaskMessage]) -> TaskMessage:
        next_task = min(
            task_list,
            key=lambda x: (np.log(x["ect"]) - x["wait"]),
        )
        task_list.remove(next_task)
        for i, task in enumerate(task_list):
            task.update(
                {
                    "wait": task["wait"]
                    + (task["ect"] / 1000 / self._num_runners)
                }
            )
            task_list[i] = task
        return next_task

    def choose_resource(self, runner_ect: Dict[int, int]) -> int:
        next_runner_id = min(runner_ect, key=runner_ect.get)
        return next_runner_id


# class HybridSJFLBAP(SchedulingPolicy):
#     def choose_task(self, task_list: List[TaskMessage]) -> TaskMessage:
#         if len(task_list) <= 10000:
#             next_task = min(
#                 task_list,
#                 key=lambda x: (np.log(x["ect"]) - x["wait"]),
#             )
#             task_list.remove(next_task)
#             for i, task in enumerate(task_list):
#                 task.update(
#                     {
#                         "wait": task["wait"]
#                         + (task["ect"] / 2000 / self._num_runners)
#                     }
#                 )
#                 task_list[i] = task
#         else:
#             next_task = task_list.pop(0)
#         return next_task

#     def choose_resource(self, runner_ect: Dict[int, int]) -> int:
#         next_runner_id = min(runner_ect, key=runner_ect.get)
#         return next_runner_id
