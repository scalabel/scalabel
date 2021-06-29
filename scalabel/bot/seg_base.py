"""Abstract interface for segmentation model."""
from abc import ABC, abstractmethod
from typing import List

from ..common.typing import NDArrayU8


class SegBase(ABC):
    """Abstract class for segmentation interface."""

    @abstractmethod
    def convert_rect_to_poly(
        self, imgs: List[NDArrayU8], bboxes: List[List[float]]
    ) -> List[List[List[float]]]:
        """Predict rectangles -> polygons.

        Rectangles have format [x, y, w, h]
        Polygons have format [[x1, y1], [x2, y2], ...]
        """
