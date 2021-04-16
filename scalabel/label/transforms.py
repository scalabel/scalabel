"""General utils functions."""

from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from skimage import measure

from .coco_typing import PolygonType
from .typing import Box2D, Poly2D

__all__ = [
    "bbox_to_box2d",
    "box2d_to_bbox",
    "mask_to_box2d",
    "mask_to_polygon",
    "poly_to_patch",
    "poly2ds_to_mask",
    "polygon_to_poly2ds",
]


def box2d_to_bbox(box_2d: Box2D) -> List[float]:
    """Convert Scalabel Box2D into COCO bbox."""
    width = box_2d.x2 - box_2d.x1 + 1
    height = box_2d.y2 - box_2d.y1 + 1
    return [box_2d.x1, box_2d.y1, width, height]


def mask_to_box2d(mask: np.ndarray) -> Box2D:
    """Convert mask into Box2D."""
    x_inds = np.nonzero(np.sum(mask, axis=0))[0]
    y_inds = np.nonzero(np.sum(mask, axis=1))[0]
    x1, x2 = int(np.min(x_inds)), int(np.max(x_inds))
    y1, y2 = int(np.min(y_inds)), int(np.max(y_inds))
    box_2d = Box2D(x1=x1, y1=y1, x2=x2, y2=y2)
    return box_2d


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """Convert mask into bbox."""
    box_2d = mask_to_box2d(mask)
    bbox = box2d_to_bbox(box_2d)
    return bbox


def bbox_to_box2d(bbox: List[float]) -> Box2D:
    """Convert COCO bbox into Scalabel Box2D."""
    assert len(bbox) == 4
    x1, y1, width, height = bbox
    x2, y2 = x1 + width - 1, y1 + height - 1
    return Box2D(x1=x1, y1=y1, x2=x2, y2=y2)


def polygon_to_poly2ds(polygon: PolygonType) -> List[Poly2D]:
    """Convert COCO polygon into Scalabel Box2Ds."""
    poly_2ds: List[Poly2D] = []
    for poly in polygon:
        point_num = len(poly) // 2
        assert 2 * point_num == len(poly)
        vertices = [[poly[2 * i], poly[2 * i + 1]] for i in range(point_num)]
        poly_2d = Poly2D(vertices=vertices, types="L" * point_num, closed=True)
        poly_2ds.append(poly_2d)
    return poly_2ds


def poly_to_patch(
    vertices: List[Tuple[float, float]],
    types: str,
    color: Tuple[float, float, float],
    closed: bool,
) -> mpatches.PathPatch:
    """Draw polygons using the Bezier curve."""
    moves = {"L": Path.LINETO, "C": Path.CURVE4}
    points = list(vertices)
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.LINETO)

    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else "none",
        edgecolor=color,
        lw=0 if closed else 1,
        alpha=1,
        antialiased=False,
        snap=True,
    )


def poly2ds_to_mask(
    shape: Tuple[int, int], poly2d: List[Poly2D]
) -> np.ndarray:
    """Converting Poly2D to mask."""
    fig = plt.figure(facecolor="0")
    fig.set_size_inches(shape[1] / fig.get_dpi(), shape[0] / fig.get_dpi())
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for poly in poly2d:
        ax.add_patch(
            poly_to_patch(
                poly.vertices,
                poly.types,
                color=(1, 1, 1),
                closed=True,
            )
        )

    fig.canvas.draw()
    mask: np.ndarray = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    mask = mask.reshape((*shape, -1))[..., 0]
    plt.close()
    return mask


def close_contour(contour: np.ndarray) -> np.ndarray:
    """Explicitly close the contour."""
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def mask_to_polygon(
    binary_mask: np.ndarray, x_1: int, y_1: int, tolerance: float = 0.5
) -> List[List[float]]:
    """Convert BitMask to polygon."""
    polygons = []
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        for i, _ in enumerate(segmentation):
            if i % 2 == 0:
                segmentation[i] = float(segmentation[i] + x_1)
            else:
                segmentation[i] = float(segmentation[i] + y_1)

        polygons.append(segmentation)

    return polygons
