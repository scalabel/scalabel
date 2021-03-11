"""Type definition for scalabel format."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from dataclasses_json import LetterCase, dataclass_json  # type: ignore


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Box2D:
    """Box 2D."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Box3D:
    """Box 3D."""

    alpha: float
    orientation: Tuple[float, float, float]
    location: Tuple[float, float, float]
    dimension: Tuple[float, float, float]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Poly2D:
    """Polygon or polyline 2D."""

    vertices: List[Tuple[float, float]]
    types: str
    closed: bool


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Label:
    """Label."""

    id: str
    manual_shape: Union[bool, None] = None
    manual_ttributes: Union[bool, None] = None
    score: Union[float, None] = None
    attributes: Union[Dict[str, str], None] = None
    box2d: Union[None, Box2D] = None
    box3d: Union[None, Box3D] = None
    poly2d: Union[None, Poly2D] = None

    def __post_init__(self) -> None:
        """Check field types."""
        self.id = str(self.id)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Frame:
    """Frame."""

    name: str
    video_name: Union[str, None] = None
    attributes: Union[None, Dict[str, str]] = None
    timestamp: Union[None, int] = None
    frame_index: Union[None, int] = None
    size: Union[None, List[int]] = None
    labels: List[Label] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check field types."""
        self.name = str(self.name)
