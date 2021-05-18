"""Type definition for scalabel format."""
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

Size = Tuple[int, int]


class Box2D(BaseModel):
    """Box 2D."""

    x1: float
    y1: float
    x2: float
    y2: float


class Box3D(BaseModel):
    """Box 3D."""

    alpha: float
    orientation: Tuple[float, float, float]
    location: Tuple[float, float, float]
    dimension: Tuple[float, float, float]


class Poly2D(BaseModel):
    """Polygon or polyline 2D."""

    vertices: List[Tuple[float, float]]
    types: str
    closed: bool


class Label(BaseModel):
    """Label."""

    id: str
    index: Optional[int] = None
    manual_shape: Optional[bool] = None
    manual_attributes: Optional[bool] = None
    score: Optional[float] = None
    attributes: Optional[Dict[str, Union[bool, int, float, str]]] = None
    category: Optional[str] = None
    box2d: Optional[Box2D]
    box3d: Optional[Box3D]
    poly2d: Optional[List[Poly2D]]

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init structure and convert the id type to string."""
        if "id" in data:
            data["id"] = str(data["id"])
        super().__init__(**data)


class ImageSize(BaseModel):
    """Define image size in config."""

    width: int
    height: int


class Intrinsics(BaseModel):
    """Camera intrinsics."""

    # focal length in (x, y)
    focal: Tuple[float, float]
    # center position in (x, y)
    center: Tuple[float, float]
    skew: float = 0
    # radial distortion parameters
    radial: Optional[Tuple[float, float, float]]
    # tangential distortion parameters
    tangential: Optional[Tuple[float, float]]


class Extrinsics(BaseModel):
    """Camera extrinsics."""

    # 3D location relative to a world origin
    location: Tuple[float, float, float]
    # 3D rotation relative to a world origin in axis-angle representation
    rotation: Tuple[float, float, float]


class Frame(BaseModel):
    """Frame."""

    name: str
    url: Optional[str]
    video_name: Optional[str] = None
    intrinsics: Optional[Intrinsics] = None
    extrinsics: Optional[Extrinsics] = None
    attributes: Optional[Dict[str, Union[str, float]]] = None
    timestamp: Optional[int] = None
    frame_index: Optional[int] = None
    size: Optional[ImageSize] = None
    labels: Optional[List[Label]] = None

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init structure and convert the id type to string."""
        if "name" in data:
            data["name"] = str(data["name"])
        super().__init__(**data)


class Category(BaseModel):
    """Define Scalabel label attributes."""

    name: str
    subcategories: Optional[List["Category"]]


Category.update_forward_refs()


class Attribute(BaseModel):
    """Define Scalabel category type."""

    name: str
    type: str
    tag: str
    values: Optional[List[str]]


class Config(BaseModel):
    """Define metadata of the dataset."""

    # optional image size info to make memory pre-allocation possible
    image_size: Optional[ImageSize]
    attributes: Optional[List[Attribute]]
    categories: List[Category]


class Dataset(BaseModel):
    """Define dataset components."""

    frames: List[Frame]
    config: Optional[Config]
