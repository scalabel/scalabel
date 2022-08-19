import numpy as np
from tqdm import tqdm
from typing import Callable, List, Type

from scalabel_bot.common.consts import ServiceMode, T
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.services.common.generic_model import GenericModel
from scalabel_bot.services.common.generic_service import GenericService
from scalabel_bot.services.dd3d import DD3D


class DD3DInference(GenericService):
    def __init__(self) -> None:
        super().__init__()
        self._name: str = "dd3d_inference"
        self._service_mode: ServiceMode = ServiceMode.INFERENCE
        self._model: GenericModel = DD3D()
        self._func: Callable[
            [TaskMessage, List[T]], List[T]
        ] = self.import_func()

    def import_func(self) -> Callable[[TaskMessage, List[T]], List[T]]:
        def inference(task: TaskMessage, data: List[T]) -> List[T]:
            output: List[T] = []
            for img in tqdm(
                np.array_split(np.array(data), len(data)),
                desc=f"Task {task['projectName']}_{task['taskId']}",
                leave=True,
                position=0,
                unit="items",
            ):
                output.extend(self._model(img))
            return output

        return inference


service_class: Type[GenericService] = DD3DInference
