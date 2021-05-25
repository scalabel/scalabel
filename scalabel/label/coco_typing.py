"""Type definitions for the COCO format."""

import sys
from typing import List, Optional, Union

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

PolygonType = List[List[float]]


class CatType(TypedDict):
    """Define types of categories in GT."""

    id: int
    name: str


class RLEType(TypedDict):
    """Defines types of polygons in GT."""

    counts: List[int]
    size: List[int]


class AnnType(TypedDict, total=False):
    """Define types of annotations in GT."""

    id: int
    image_id: int
    category_id: int
    iscrowd: int
    ignore: int
    instance_id: Optional[int]
    scalabel_id: Optional[str]
    score: Optional[float]
    bbox: Optional[List[float]]
    area: Optional[float]
    segmentation: Optional[Union[PolygonType, RLEType]]


class ImgType(TypedDict, total=False):
    """Define types of images in GT."""

    id: int
    file_name: str
    height: int
    width: int
    coco_url: Optional[str]
    video_id: Optional[int]
    frame_id: Optional[int]


class VidType(TypedDict):
    """Define types of videos in GT."""

    id: int
    name: str


class GtType(TypedDict, total=False):
    """Define types of the GT in COCO format."""

    categories: List[CatType]
    annotations: List[AnnType]
    images: List[ImgType]
    type: str
    videos: Optional[List[VidType]]


class PredType(TypedDict):
    """Define input prediction type."""

    category: str
    score: float
    name: str
    bbox: List[float]
    image_id: int
    category_id: int


class PnpSegType(TypedDict, total=False):
    """Define types of segment_info for panoptic segmentation."""

    id: int
    category_id: int
    area: float
    bbox: List[float]
    iscrowd: int


class PnpAnnType(TypedDict, total=False):
    """Define types of annotation for panoptic segmentation."""

    image_id: int
    file_name: str
    segments_info: List[PnpSegType]


class PnPCatType(TypedDict):
    """Define types of categories in GT of panoptic segmentation."""

    id: int
    name: str
    supercategory: str
    isthing: int
    color: str


class PnPGtType(TypedDict, total=False):
    """Define types of the GT in COCO panoptic segmentaton format."""

    categories: List[PnPCatType]
    annotations: List[PnpAnnType]
    images: List[ImgType]
