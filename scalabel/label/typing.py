"""Type definition for scalabel format."""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

DictStrAny = Dict[str, Any]  # type: ignore[misc]


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
    manual_shape: Optional[bool] = None
    manual_attributes: Optional[bool] = None
    score: Optional[float] = None
    attributes: Optional[Dict[str, Union[str, float, bool]]] = None
    box_2d: Optional[Box2D]
    box_3d: Optional[Box3D]
    poly_2d: Optional[Poly2D]

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init structure and convert the id type to string."""
        if "id" in data:
            data["id"] = str(data["id"])
        super().__init__(**data)


class Frame(BaseModel):
    """Frame."""

    name: str
    video_name: Optional[str] = None
    attributes: Optional[Dict[str, Union[str, float]]] = None
    timestamp: Optional[int] = None
    frame_index: Optional[int] = None
    size: Optional[List[int]] = None
    labels: List[Label]

    def __init__(self, **data: Any) -> None:  # type: ignore
        """Init structure and convert the id type to string."""
        if "name" in data:
            data["name"] = str(data["name"])
        super().__init__(**data)
