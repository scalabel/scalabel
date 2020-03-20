""" Abstract interface with the polyrnn model """
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class PolyrnnBase(ABC):
    """ Abstract class for polyrnn interface """

    def __init__(self):
        pass

    @abstractmethod
    def predict_from_rect(self, crop_img):
        """ predict rect -> poly """

    @abstractmethod
    def bbox_to_crop(self, img, bbox):
        """ create square crop around expanded bbox """

    @abstractmethod
    def rescale_output(self, preds: List[np.ndarray], crop_dict) -> List[List[float]]:
        """ undo the cropping transform to get original output coords """
