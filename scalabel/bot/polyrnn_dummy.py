""" Dummy interface with the polyrnn model """
from typing import List
import numpy as np
from .polyrnn_base import PolyrnnBase


class PolyrnnDummy(PolyrnnBase):
    """ Class for polyrnn dummy interface """

    def __init__(self) -> None:
        # init method for consistent API
        return

    def predict_rect_to_poly(
            self, img: np.ndarray, bbox: List[float]) -> List[List[float]]:
        """ predict rect -> dummy polygon """
        x0, y0, w, h = bbox

        # create diamond inscribed in rectangle
        preds = np.array([[w/2.0, 0], [w, h/2.0], [w/2.0, h], [0, h/2.0]])
        start_point = np.array([x0, y0])

        preds = start_point + preds

        output: List[List[float]] = preds.tolist()
        return output
