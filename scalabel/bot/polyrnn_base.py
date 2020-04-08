""" Abstract interface with the polyrnn model """
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class PolyrnnBase(ABC):
    """ Abstract class for polyrnn interface """
    @abstractmethod
    def predict_rect_to_poly(
            self,
            imgs: List[np.ndarray],
            bboxes: List[List[float]]) -> List[List[List[float]]]:
        """
        Predict rectangles -> polygons
        Rectangles have format [x, y, w, h]
        Polygons have format [[x1, y1], [x2, y2], ...]
        """
        