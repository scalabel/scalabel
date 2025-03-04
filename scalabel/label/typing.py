"""Type definition for scalabel format."""
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

Size = Tuple[int, int]


class Box2D(BaseModel):
    """Box 2D."""

    x1: float= None
    y1: float = None
    x2: float = None
    y2: float = None


class Box3D(BaseModel):
    """Box 3D."""

    alpha: float
    orientation: Tuple[float, float, float] = None
    location: Tuple[float, float, float] = None
    dimension: Tuple[float, float, float] = None


class Poly2D(BaseModel):
    """Polygon or polyline 2D."""

    vertices: List[Tuple[float, float]] = None
    types: str = None
    closed: bool = None


class RLE(BaseModel):
    """Bitmask in RLE format."""

    counts: str = None
    size: Tuple[int, int] = None


class Node(BaseModel):
    """Node of a graph."""

    # 2D or 3D coordinates.
    # in 2D: (x, y), x horizontal, y vertical, (0, 0) top left corner
    location: Union[Tuple[float, float], Tuple[float, float, float]] = None
    category: str = None
    visibility: Optional[str] = None
    type: Optional[str] = None
    score: Optional[float] = None
    id: str = None


class Edge(BaseModel):
    """Edge of a graph."""

    source: str = None
    target: str = None
    type: Optional[str] = None


class Graph(BaseModel):
    """Graph."""

    nodes: List[Node] = None
    edges: List[Edge] = None
    type: Optional[str] = None


class Label(BaseModel):
    """Label."""

    id: str
    index: Optional[int] = None
    manualShape: Optional[bool] = None
    manualAttributes: Optional[bool] = None
    score: Optional[float] = None
    attributes: Optional[Dict[str, Union[bool, int, float, str]]] = None
    category: Optional[str] = None
    box2d: Optional[Box2D] = None
    box3d: Optional[Box3D] = None
    poly2d: Optional[List[Poly2D]] = None
    rle: Optional[RLE] = None
    graph: Optional[Graph] = None

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init structure and convert the id type to string."""
        if "id" in data:
            data["id"] = str(data["id"])
        super().__init__(**data)


class ImageSize(BaseModel):
    """Define image size in config."""

    width: int = None
    height: int = None


class Intrinsics(BaseModel):
    """Camera intrinsics."""

    # focal length in (x, y)
    focal: Tuple[float, float] = None
    # center position in (x, y)
    center: Tuple[float, float] = None
    skew: float = 0
    # radial distortion parameters
    radial: Optional[Tuple[float, float, float]] = None
    # tangential distortion parameters
    tangential: Optional[Tuple[float, float]] = None


class Extrinsics(BaseModel):
    """Camera extrinsics."""

    # 3D location relative to a world origin
    location: Tuple[float, float, float] = None
    # 3D rotation relative to a world origin in axis-angle representation
    rotation: Tuple[float, float, float] = None


class Frame(BaseModel):
    """Frame."""

    name: str = None
    url: Optional[str] = None
    videoName: Optional[str] = None
    intrinsics: Optional[Intrinsics] = None
    extrinsics: Optional[Extrinsics] = None
    attributes: Optional[Dict[str, Union[str, float]]] = None
    timestamp: Optional[int] = None
    frameIndex: Optional[int] = None
    size: Optional[ImageSize] = None
    labels: Optional[List[Label]] = None

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init structure and convert the id type to string."""
        if "name" in data:
            data["name"] = str(data["name"])
        super().__init__(**data)


class Category(BaseModel):
    """Define Scalabel label attributes."""

    name: str = None
    subcategories: Optional[List["Category"]] =None
    isThing: Optional[bool] = None  # for panoptic segmentation
    color: Optional[Tuple[float, float, float]] = None


Category.update_forward_refs()


class Attribute(BaseModel):
    """Define Scalabel attribute type."""

    name: str = None
    type: str = None
    tag: Optional[str] = None
    tagPrefix: Optional[str] = None
    tagSuffixes: Optional[List[str]] = None
    values: Optional[List[str]] = None


class Config(BaseModel):
    """Define metadata of the dataset."""

    # optional image size info to make memory pre-allocation possible
    imageSize: Optional[ImageSize] = None
    attributes: Optional[List[Attribute]]  = None
    categories: List[Category] = None
    poseSigmas: Optional[List[float]]  = None


class FrameGroup(Frame):
    """Define group of frames and shared attributes."""

    frames: List[str] = None


class Dataset(BaseModel):
    """Define dataset components."""

    frames: List[Frame] = None
    groups: Optional[List[FrameGroup]] = None
    config: Optional[Config] = None
