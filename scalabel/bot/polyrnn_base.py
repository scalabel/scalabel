""" Abstract interface with the polyrnn model """
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class PolyrnnBase(ABC):
    """ Abstract class for polyrnn interface """
    @abstractmethod
    def predict_rect_to_poly(
            self, img: np.ndarray, bbox: List[float]) -> List[List[float]]:
        """ predict rect -> poly """
        