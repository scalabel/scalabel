from tqdm import tqdm
from typing import Callable, List, Type

from scalabel_bot.common.consts import ServiceMode, T
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.services.common.generic_model import GenericModel
from scalabel_bot.services.common.generic_service import GenericService
from scalabel_bot.services.fsdet import FSDET


class FSDETInference(GenericService):
    def __init__(self) -> None:
        super().__init__()
        self._name: str = "fsdet_inference"
        self._service_mode: ServiceMode = ServiceMode.INFERENCE
        self._model: GenericModel = FSDET()
        self._func: Callable[
            [TaskMessage, List[T]], List[T]
        ] = self.import_func()

    def import_func(self) -> Callable[[TaskMessage, List[T]], List[T]]:
        def inference(task: TaskMessage, data: List[T]) -> List[T]:
            output: List[T] = []
            for img in tqdm(
                data,
                desc=f"Task {task['taskId']}",
                leave=True,
                position=0,
                unit="items",
            ):
                predictions: List[T] = self._model(img)
                output.extend(predictions)
            return output

        return inference


service_class: Type[GenericService] = FSDETInference
