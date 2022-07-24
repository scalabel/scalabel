import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple  # pylint: disable=unused-import
from threading import Thread
from time import sleep
from torch.multiprocessing import (  # pylint: disable=unused-import
    Event,
    Process,
    Queue,
)

from scalabel_bot.common.consts import State, Timers
from scalabel_bot.common.func import cantor_pairing
from scalabel_bot.common.logger import logger
from scalabel_bot.profiling.timer import timer
from scalabel_bot.runner.runner import Runner


class Policy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_next(self, runners: Dict[int, Runner], runner_id: int) -> int:
        pass


class RoundRobinPolicy:
    @timer(Timers.THREAD_TIMER)
    def select_next(self, runner_ect: Dict[int, int], runner_id: int) -> int:
        next_runner_id = (runner_id + 1) % len(runner_ect)
        return list(runner_ect.keys())[next_runner_id]


class LoadBalancingPolicy:
    @timer(Timers.THREAD_TIMER)
    def select_next(self, runner_ect: Dict[int, int]) -> int:
        next_runner_id = min(runner_ect, key=runner_ect.get)
        return next_runner_id


class Exp3Policy:
    def __init__(self) -> None:
        super().__init__()
        self.num_resources: int = 0
        self.num_tasks: int = 0
        self.gamma: float = 0.7  # egalitarian factor
        self.weights: List[float] = []

    # pick an index from the given list of floats proportionally
    # to the size of the entry (i.e. normalize to a probability
    # distribution and draw according to the probabilities).
    def draw(self, weights):
        choice = random.uniform(0, sum(weights))
        choiceIndex = 0

        for weight in weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1

    # Normalize a list of floats to a probability distribution.  Gamma is an
    # egalitarianism factor, which tempers the distribtution toward being uniform as
    # it grows from zero to one.
    def distr(self, weights, gamma=0.0):
        theSum = float(sum(weights))
        return tuple(
            (1.0 - gamma) * (w / theSum) + (gamma / len(weights))
            for w in weights
        )

    def allocate_tasks(self, _tasks, _action):
        task_allocation = []
        while len(_action):
            task_group = []
            resources_used = 0
            _tasks_copy = _tasks.copy()
            _action_copy = _action.copy()
            for index, task in enumerate(_tasks_copy):
                if (
                    resources_used + (1 / _action_copy[index])
                    > self.num_resources
                ):
                    continue
                resources_used += 1 / _action_copy[index]
                _tasks.remove(task)
                task = (task[0], task[1] * _action_copy[index])
                task_group.append(task)
                _action.remove(_action_copy[index])
                if resources_used == self.num_resources:
                    break
            max_exec_time = max(task_group, key=lambda k: k[1])[1]
            task_allocation.append((task_group, max_exec_time))
        return task_allocation

    def calc_reward(self, allocation):
        total_exec_time = sum(exec_time for _, exec_time in allocation)
        reward = math.pow((1 / total_exec_time), 2)
        return reward

    def calc_total_time(self, tasks, action):
        allocation = self.allocate_tasks(tasks, action)
        return sum(exec_time for _, exec_time in allocation)

    # perform the exp3 algorithm.
    def run(self, tasks, gamma):
        num_actions = self.calc_num_actions()
        weights = [1.0] * num_actions
        task_shard = self.task_sharding()

        while True:
            prob_distr = self.distr(weights, gamma)
            choice = self.draw(prob_distr)
            # selects the action based
            action = [
                task_shard[
                    (choice // pow(self.num_resources, j)) % self.num_resources
                ]
                for j in range(self.num_tasks)
            ]
            allocation = self.allocate_tasks(tasks.copy(), action)
            reward = self.calc_reward(allocation)
            est_reward = 1.0 * reward / prob_distr[choice]
            # update the weights based on estimated reward
            weights[choice] *= math.exp(est_reward * gamma)
            weights = self.normalise(weights)

            yield choice, reward, weights

    def normalise(self, weights):
        total_weight = sum(weights)
        norm_weights = [weight / total_weight for weight in weights]
        return norm_weights

    def calc_num_actions(self):
        return int(math.pow(self.num_resources, self.num_tasks))

    # list of possible ways to partition a task
    def task_sharding(self):
        return [1 / i for i in range(1, self.num_resources + 1)]

    def find_best_action(self, weights, task_shard):
        index = weights.index(max(weights))
        action = [
            task_shard[
                (index // pow(self.num_resources, j)) % self.num_resources
            ]
            for j in range(self.num_tasks)
        ]
        return action


class RunnerScheduler(Process):
    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        runner_status: Dict[int, State],
        runner_status_queue: "Queue[Tuple[int, int, State]]",
        runner_ect: Dict[int, int],
        runner_ect_queue: "Queue[Tuple[int, int, int]]",
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop_run: Event = Event()
        self._policy: RoundRobinPolicy = RoundRobinPolicy()
        self._runner_status: Dict[int, State] = runner_status
        self._runner_status_queue: "Queue[Tuple[int, int, State]]" = (
            runner_status_queue
        )
        self._runner_ect_queue: "Queue[Tuple[int, int, int]]" = (
            runner_ect_queue
        )
        self._runner_ect: Dict[int, int] = runner_ect
        self._curr_runner_id: int = 0

    def run(self) -> None:
        while not self._stop_run.is_set():
            device, runner_id, status = self._runner_status_queue.get()
            logger.debug(f"{self._name}: Runner {device}-{runner_id} {status}")
            self._runner_status[cantor_pairing(device, runner_id)] = status

    def _update_ect(self):
        while not self._stop_run.is_set():
            device, runner_id, ect = self._runner_ect_queue.get()
            self._runner_ect[cantor_pairing(device, runner_id)] = ect

    @timer(Timers.THREAD_TIMER)
    def schedule(self) -> int:
        while not self._runner_ect_queue.empty():
            device, runner_id, ect = self._runner_ect_queue.get()
            self._runner_ect[cantor_pairing(device, runner_id)] = ect
        if isinstance(self._policy, LoadBalancingPolicy):
            self._curr_runner_id = self._policy.select_next(self._runner_ect)
        elif isinstance(self._policy, RoundRobinPolicy):
            self._curr_runner_id = self._policy.select_next(
                self._runner_ect, self._curr_runner_id
            )
        return self._curr_runner_id

    @timer(Timers.THREAD_TIMER)
    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name}: stopping...")
        self._stop_run.set()
        logger.debug(f"{self._name}: stopped!")

    @property
    def runner_status(self) -> Dict[int, State]:
        return self._runner_status
