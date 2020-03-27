""" Interface with the polyrnn model """
from typing import List
import os
import warnings
import numpy as np
from Tool.tool import Tool  # type: ignore
from .polyrnn_base import PolyrnnBase


class PolyrnnInterface(PolyrnnBase):
    """Class to interface with the PolygonRNN model"""
    def __init__(self, home: str) -> None:
        exp = os.path.join(home, 'experimental', 'fast-seg-label',
                           'polyrnn', 'Experiments', 'tool.json')
        net = os.path.join(home, 'experimental_models',
                           'ggnn_epoch5_step14000.pth')
        self.tool = Tool(exp, net)

    def predict_rect_to_poly(
            self, img: np.ndarray, bbox: List[float]) -> List[List[float]]:
        # marshal into polyrnn codebase's format
        instance = {
            'img': img,
            'bbox': bbox
        }
        component = {
            'poly': np.array([[-1., -1.]])
        }
        instance = self.tool.data_loader.prepare_component(instance, component)

        # ignore deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.tool.annotation(instance)

        output: List[List[float]] = preds[0]
        return output
    