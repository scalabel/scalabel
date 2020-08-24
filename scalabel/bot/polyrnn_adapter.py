"""Interface with the segmentation model."""
from typing import List
import os
import warnings
import numpy as np
from Tool.tool import Tool  # pylint: disable=import-error
from .seg_base import SegBase


class PolyrnnAdapter(SegBase):
    """Class to interface with polyrnn model."""

    def __init__(self, home: str) -> None:
        """Initialize."""
        model_dir = os.path.join(home, "experimental", "fast-seg-label")
        exp = os.path.join(
            model_dir, "polyrnn_scalabel", "Experiments", "tool.json"
        )
        net = os.path.join(
            model_dir, "model_weights", "ggnn_epoch5_step14000.pth"
        )
        self.tool = Tool(exp, net)

    def convert_rect_to_poly(
        self, imgs: List[np.ndarray], bboxes: List[List[float]]
    ) -> List[List[List[float]]]:
        """Convert rectangles to polygons."""
        # marshal into segmentation codebase's format
        inputs = []
        for (img, bbox) in zip(imgs, bboxes):
            instance = {"img": img, "bbox": bbox}
            component = {"poly": np.array([[-1.0, -1.0]])}
            instance = self.tool.preprocess_instance(instance, component)
            inputs.append(instance)

        # ignore deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds: List[List[List[float]]] = self.tool.annotate_batch(inputs)
        return preds
