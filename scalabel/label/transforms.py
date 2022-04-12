"""General utils functions."""

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from nanoid import generate  # type: ignore
from pycocotools import mask as mask_utils  # type: ignore

from ..common.typing import NDArrayU8
from .coco_typing import CatType, PolygonType, RLEType
from .typing import RLE, Box2D, Config, Edge, Graph, ImageSize, Node, Poly2D
from .utils import get_leaf_categories

__all__ = [
    "get_coco_categories",
    "bbox_to_box2d",
    "box2d_to_bbox",
    "frame_to_masks",
    "frame_to_rles",
    "mask_to_box2d",
    "mask_to_rle",
    "poly_to_patch",
    "poly2ds_to_mask",
    "polygon_to_poly2ds",
    "keypoints_to_nodes",
    "rle_to_box2d",
    "rle_to_mask",
    "coco_rle_to_rle",
    "graph_to_keypoints",
]


def get_coco_categories(config: Config) -> List[CatType]:
    """Get CatType categories for saving these in COCO format annotations."""
    categories = get_leaf_categories(config.categories)
    result = [
        CatType(id=i + 1, name=category.name)
        for i, category in enumerate(categories)
    ]
    return result


def box2d_to_bbox(box2d: Box2D) -> List[float]:
    """Convert Scalabel Box2D into COCO bbox."""
    width = box2d.x2 - box2d.x1 + 1
    height = box2d.y2 - box2d.y1 + 1
    return [box2d.x1, box2d.y1, width, height]


def mask_to_box2d(mask: NDArrayU8) -> Box2D:
    """Convert mask into Box2D."""
    x_inds = np.nonzero(np.sum(mask, axis=0))[0]
    y_inds = np.nonzero(np.sum(mask, axis=1))[0]
    x1, x2 = int(np.min(x_inds)), int(np.max(x_inds))
    y1, y2 = int(np.min(y_inds)), int(np.max(y_inds))
    box2d = Box2D(x1=x1, y1=y1, x2=x2, y2=y2)
    return box2d


def mask_to_bbox(mask: NDArrayU8) -> List[float]:
    """Convert mask into bbox."""
    box2d = mask_to_box2d(mask)
    bbox = box2d_to_bbox(box2d)
    return bbox


def xyxy_to_box2d(x1: float, y1: float, x2: float, y2: float) -> Box2D:
    """Transform xyxy box (not incl xy_2) to scalabel (incl xy_2 in box)."""
    return Box2D(x1=x1, y1=y1, x2=x2 - 1, y2=y2 - 1)


def box2d_to_xyxy(box2d: Box2D) -> List[float]:
    """Transform scalabel box (include xy_2 in box) to xyxy (not incl xy_2)."""
    return [box2d.x1, box2d.y1, box2d.x2 + 1, box2d.y2 + 1]


def bbox_to_box2d(bbox: List[float]) -> Box2D:
    """Convert COCO bbox into Scalabel Box2D."""
    assert len(bbox) == 4
    x1, y1, width, height = bbox
    x2, y2 = x1 + width - 1, y1 + height - 1
    return Box2D(x1=x1, y1=y1, x2=x2, y2=y2)


def polygon_to_poly2ds(polygon: PolygonType) -> List[Poly2D]:
    """Convert COCO polygon into Scalabel Box2Ds."""
    poly2ds: List[Poly2D] = []
    for poly in polygon:
        point_num = len(poly) // 2
        assert 2 * point_num == len(poly)
        vertices = [[poly[2 * i], poly[2 * i + 1]] for i in range(point_num)]
        poly2d = Poly2D(vertices=vertices, types="L" * point_num, closed=True)
        poly2ds.append(poly2d)
    return poly2ds


def coco_rle_to_rle(rle: RLEType) -> RLE:
    """Convert COCO RLE into Scalabel RLE."""
    size = rle["size"]
    if isinstance(rle["counts"], str):
        counts = rle["counts"]
    elif isinstance(rle["counts"], list):
        counts = mask_utils.frPyObjects(rle, *size)["counts"].decode("utf-8")
    else:
        counts = rle["counts"].decode("utf-8")
    return RLE(counts=counts, size=size)


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


def poly2ds_to_mask(shape: ImageSize, poly2d: List[Poly2D]) -> NDArrayU8:
    """Converting Poly2D to mask."""
    fig = plt.figure(facecolor="0")
    fig.set_size_inches(
        shape.width / fig.get_dpi(), shape.height / fig.get_dpi()
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, shape.width)
    ax.set_ylim(0, shape.height)
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for poly in poly2d:
        ax.add_patch(
            poly_to_patch(
                poly.vertices,
                poly.types,
                color=(1, 1, 1),
                closed=poly.closed,
            )
        )

    fig.canvas.draw()
    mask: NDArrayU8 = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    mask = mask.reshape((shape.height, shape.width, -1))[..., 0]
    plt.close()
    return mask


def frame_to_masks(
    shape: ImageSize, poly2ds: List[List[Poly2D]]
) -> List[NDArrayU8]:
    """Converting a frame of poly2ds to masks/bitmasks. Removes overlaps."""
    height, width = shape.height, shape.width
    fig = plt.figure(facecolor="0")
    fig.set_size_inches((width / fig.get_dpi()), height / fig.get_dpi())
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    for i, poly2d in enumerate(poly2ds):
        for poly in poly2d:
            ax.add_patch(
                poly_to_patch(
                    poly.vertices,
                    poly.types,
                    # (0, 0, 0) for the background
                    color=(
                        ((i + 1) >> 8) / 255.0,
                        ((i + 1) % 255) / 255.0,
                        0.0,
                    ),
                    closed=poly.closed,
                )
            )

    fig.canvas.draw()
    out: NDArrayU8 = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    out = out.reshape((height, width, -1)).astype(np.int32)
    out = (out[..., 0] << 8) + out[..., 1]
    plt.close()

    masks = []
    for i, _ in enumerate(poly2ds):
        mask: NDArrayU8 = np.zeros([height, width, 1], dtype=np.uint8)
        mask[out == i + 1] = 255
        masks.append(mask.squeeze(2))
    return masks


def mask_to_rle(mask: NDArrayU8) -> RLE:
    """Converting mask to RLE format."""
    assert 2 <= len(mask.shape) <= 3
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    rle = mask_utils.encode(np.array(mask, order="F", dtype="uint8"))[0]
    return RLE(counts=rle["counts"].decode("utf-8"), size=rle["size"])


def frame_to_rles(
    shape: ImageSize, poly2ds: List[List[Poly2D]], no_overlap: bool = True
) -> List[RLE]:
    """Converting frame of Poly2Ds to RLEs."""
    if no_overlap:
        masks = frame_to_masks(shape, poly2ds)
    else:
        masks = [poly2ds_to_mask(shape, poly2d) for poly2d in poly2ds]
    return [mask_to_rle(mask) for mask in masks]


def rle_to_mask(rle: RLE) -> NDArrayU8:
    """Converting RLE to mask."""
    return mask_utils.decode(dict(rle))  # type: ignore


def rle_to_box2d(rle: RLE) -> Box2D:
    """Converting RLE to Box2D."""
    bbox = mask_utils.toBbox(rle.dict()).tolist()
    return bbox_to_box2d(bbox)


def keypoints_to_nodes(
    kpts: List[float], cats: Optional[List[str]] = None
) -> List[Node]:
    """Converting COCO keypoints to list of Nodes."""
    assert len(kpts) % 3 == 0
    if cats is None:
        cats = ["coco_kpt"] * (len(kpts) // 3)
    return [
        Node(
            location=(kpts[i], kpts[i + 1]),
            category=cats[i // 3],
            id=generate(size=16),
            score=kpts[i + 2],
        )
        for i in range(0, len(kpts), 3)
    ]


def nodes_to_edges(
    nodes: List[Node], edge_map: Dict[int, Tuple[List[int], str]]
) -> List[Edge]:
    """Converting list of Nodes to list of Edges using an edge map.

    edge_map is a mapping from source node index to a tuple consisting of a
    list of target nodes' indices and edge type.
    """
    edges = []
    for edge_idx in edge_map.keys():
        conns, etype = edge_map[edge_idx]
        for conn in conns:
            edges.append(
                Edge(
                    source=nodes[edge_idx].id,
                    target=nodes[conn].id,
                    type=etype,
                )
            )
    return edges


def graph_to_keypoints(graph: Graph) -> List[float]:
    """Converting Graph to COCO keypoints."""
    keypoints = []
    for node in graph.nodes:
        c3 = 0.0
        if node.score is not None:
            c3 = node.score
        else:
            if graph.type is not None:
                if graph.type.startswith("Pose2D"):
                    if node.visibility == "V":
                        c3 = 2.0
                    elif node.visibility == "N":
                        c3 = 1.0
        keypoints.extend([node.location[0], node.location[1], c3])
    return keypoints
