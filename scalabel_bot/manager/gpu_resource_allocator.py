# -*- coding: utf-8 -*-
"""PipeSwitch GPU Resource Allocator

This module queries available GPUs and allocates GPU resources
for the PipeSwitch Manager.

Todo:
    * None
"""

import os
from typing import OrderedDict, List
from gpustat import GPUStat, GPUStatCollection  # type: ignore
import torch

from scalabel_bot.common.consts import Timers
from scalabel_bot.common.exceptions import GPUError

from scalabel.common.logger import logger
from scalabel_bot.profiling.timer import timer


class GPUResourceAllocator(object):
    """GPU Resource Allocator class.

    It has two main functions:
        - Queries available GPUs and reserves them.
        - Checks if each available GPU can be utilized by PyTorch.

    Attributes:
        gpus (`OrderedDict[int, GPUStat]`):
            Dictionary of all GPUs in the system regardless of availability.
    """

    @timer(Timers.PERF_COUNTER)
    def __init__(self) -> None:
        self._name: str = self.__class__.__name__
        self._gpus: OrderedDict[int, GPUStat] = self._get_gpus()
        self._cuda_init()

    @timer(Timers.PERF_COUNTER)
    def reserve_gpus(self, num_gpus: int, gpu_ids: List[int]) -> List[int]:
        """Reserves set amount of GPUs.

        Args:
            num_gpus (int, optional):
                Total number of GPUs to reserve. Defaults to 0.

        Returns:
            List[int]: List of IDs of reserved GPUs.
        """
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            logger.warning(
                f"{self._name}: CUDA_VISIBLE_DEVICES is already set"
            )
            available_gpus = [
                int(gpu)
                for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            ]
            return available_gpus

        free_gpus: List[int] = self._get_free_gpus()
        if num_gpus == 0:
            num_gpus = len(free_gpus)
        if num_gpus > len(free_gpus):
            logger.error(
                f"{self._name}: Unable to acquire {num_gpus} GPUs, there are"
                f" only {len(free_gpus)} available."
            )
            raise GPUError

        selected_gpus: List[int] = []
        if gpu_ids != []:
            for gpu_id in free_gpus:
                if gpu_id in gpu_ids:
                    selected_gpus.append(gpu_id)
        else:
            selected_gpus = free_gpus[:num_gpus]
        gpu_str: str = ",".join([str(i) for i in selected_gpus])
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        logger.debug(f"{self._name}: Acquiring GPUs:" f" {gpu_str}")
        for gpu_id in selected_gpus:
            self._check_gpu(gpu_id)
        return selected_gpus

    @timer(Timers.PERF_COUNTER)
    def warmup_gpus(self, gpus: List[int]) -> None:
        """Warmup GPUs by running a dummy PyTorch function."""
        for gpu_id in gpus:
            torch.randn(1024, device=gpu_id)

    @timer(Timers.PERF_COUNTER)
    def release_gpus(self) -> None:
        """Release all reserved GPUS."""
        if (
            "CUDA_VISIBLE_DEVICES" not in os.environ
            or os.environ["CUDA_VISIBLE_DEVICES"] == ""
        ):
            return
        logger.warning(
            f"{self._name}: Releasing all GPUs:"
            f" {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def _cuda_init(self) -> None:
        """Checks if available GPUs are visible by PyTorch.

        Raises:
            `AssertionError`: If CUDA is not available.

            `AssertionError`: If the number of GPUs visible by PyTorch
                is not equal to the total number of available GPUs.
        """
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if not torch.cuda.is_available():
            logger.error(f"{self._name}: CUDA is not available")
            raise GPUError
        if len(self._gpus) < 1 or torch.cuda.device_count() < 1:
            logger.error(f"{self._name}: No GPUs available")
            raise GPUError

    def _get_gpus(self) -> OrderedDict[int, GPUStat]:
        """Uses gpustat to query all GPUs in the system.

        Returns:
            `OrderedDict[int, GPUStat]`:
                A dictionary with GPU id as key and GPU stats as value.
        """
        stats: GPUStatCollection = GPUStatCollection.new_query()
        gpus: OrderedDict[int, GPUStat] = OrderedDict()
        for gpu in stats:
            gpus[gpu["index"]] = gpu
        return gpus

    @timer(Timers.PERF_COUNTER)
    def _get_free_gpus(self) -> List[int]:
        """Query available GPUs.

        Returns:
            List[int]: List of available GPU ids.
        """
        return [
            gpu_id
            for gpu_id, gpu in self._gpus.items()
            if (gpu["memory.total"] - gpu["memory.used"]) >= 5000
        ]

    def _check_gpu(self, gpu_id: int) -> None:
        """Checks if a GPU can be utilized by PyTorch.

        Args:
            gpu_id (int): GPU id.

        """
        device: torch.device = torch.device(gpu_id)
        x_train: torch.Tensor = torch.FloatTensor([0.0, 1.0, 2.0]).to(device)
        if not x_train.is_cuda:
            logger.error(
                f"{self._name}: GPU {gpu_id} cannot be utilised by PyTorch"
            )
