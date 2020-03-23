""" Abstract interface with the polyrnn model """
from abc import ABC, abstractmethod
from typing import List, NamedTuple
import numpy as np


class CropData(NamedTuple):
    """ Class that stores struct-style crop data """
    img: np.ndarray
    scale_factor: float
    start: np.ndarray


class PolyrnnBase(ABC):
    """ Abstract class for polyrnn interface """
    @abstractmethod
    def predict_from_rect(self, crop_img: np.ndarray) -> np.ndarray:
        """ predict rect -> poly """

    @abstractmethod
    def bbox_to_crop(self, img: np.ndarray, bbox: List[float]) -> CropData:
        """ create square crop around expanded bbox """

    @abstractmethod
    def rescale_output(self, preds: np.ndarray,
                       crop_dict: CropData) -> List[List[float]]:
        """ undo the cropping transform to get original output coords """
