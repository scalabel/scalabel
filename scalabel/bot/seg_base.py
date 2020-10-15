"""Abstract interface for segmentation model."""
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class SegBase(ABC):
    """Abstract class for segmentation interface."""

    @abstractmethod
    def convert_rect_to_poly(
        self, imgs: List[np.ndarray], bboxes: List[List[float]]
    ) -> List[List[List[float]]]:
        """Predict rectangles -> polygons.

        Rectangles have format [x, y, w, h]
        Polygons have format [[x1, y1], [x2, y2], ...]
        """
