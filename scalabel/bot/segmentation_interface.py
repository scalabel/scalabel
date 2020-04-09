""" Interface with the segmentation model """
from typing import List
import os
import warnings
import numpy as np
from Tool.tool import Tool  # type: ignore
from .segmentation_base import SegmentationBase


class SegmentationInterface(SegmentationBase):
    """Class to interface with segmentation model"""
    def __init__(self, home: str) -> None:
        exp = os.path.join(home, 'experimental', 'fast-seg-label',
                           'polyrnn_scalabel', 'Experiments', 'tool.json')
        net = os.path.join(home, 'experimental_models',
                           'ggnn_epoch5_step14000.pth')
        self.tool = Tool(exp, net)

    def predict_rect_to_poly(
            self,
            imgs: List[np.ndarray],
            bboxes: List[List[float]]) -> List[List[List[float]]]:
        # marshal into segmentation codebase's format
        inputs = []
        for (img, bbox) in zip(imgs, bboxes):
            instance = {
                'img': img,
                'bbox': bbox
            }
            component = {
                'poly': np.array([[-1., -1.]])
            }
            instance = self.tool.preprocess_instance(instance, component)
            inputs.append(instance)

        # ignore deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds: List[List[List[float]]] = self.tool.annotate_batch(inputs)
        return preds
