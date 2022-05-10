"""Dummy interface with the segmentation model."""
from typing import List

import numpy as np

from ..common.typing import NDArrayF64, NDArrayU8
from .seg_base import SegBase


class SegDummy(SegBase):
    """Dummy segmentation interface."""

    def __init__(self, _home: str) -> None:
        """Initialize."""
        # init method for consistent API
        return

    def convert_rect_to_poly(
        self, imgs: List[NDArrayU8], bboxes: List[List[float]]
    ) -> List[List[List[float]]]:
        """Predict rectangles -> dummy polygons."""
        output: List[List[List[float]]] = []
        for bbox in bboxes:
            x0, y0, w, h = bbox
            # create diamond inscribed in rectangle
            preds: NDArrayF64 = np.array(
                [[w / 2.0, 0], [w, h / 2.0], [w / 2.0, h], [0, h / 2.0]],
                dtype=np.float64,
            )
            preds += np.array([x0, y0])
            output.append(preds.tolist())
        return output
