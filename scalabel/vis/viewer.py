"""An offline label visualizer for Scalable file.

Works for 2D / 3D bounding box, segmentation masks, etc.
"""

import io
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from ..common.typing import NDArrayF64, NDArrayU8
from ..label.typing import Frame, Label
from ..label.utils import check_crowd
from .helper import (
    GenBoxFunc,
    gen_2d_rect,
    gen_3d_cube,
    poly2patch,
    random_color,
)


@dataclass
class UIConfig:
    """Visualizer UI's config class."""

    # Parameters for UI.
    height: int
    width: int
    scale: float
    dpi: int
    font: FontProperties
    default_category: str

    def __init__(
        self,
        height: int = 720,
        width: int = 1280,
        scale: float = 1.0,
        dpi: int = 80,
        default_category: str = "Car",
    ) -> None:
        """Initialize with default values."""
        self.dpi = dpi
        self.scale = scale
        self.height = height
        self.width = width
        self.default_category = default_category
        self.font = FontProperties()
        self.font.set_family(["sans-serif", "monospace"])
        self.font.set_weight("bold")
        self.font.set_size(int(18 * self.scale))


@dataclass
class DisplayConfig:
    """Visualizer display's config class."""

    # Parameters for the display.
    show_ctrl_points: bool
    show_tags: bool
    ctrl_point_size: float

    def __init__(
        self,
        show_ctrl_points: bool = False,
        show_tags: bool = True,
        ctrl_points_size: float = 2.0,
    ) -> None:
        """Initialize with default values."""
        self.show_ctrl_points = show_ctrl_points
        self.show_tags = show_tags
        self.ctrl_point_size = ctrl_points_size


class LabelViewer:
    """Visualize 2D and 3D bounding boxes."""

    def __init__(self, ui_cfg: UIConfig, display_cfg: DisplayConfig) -> None:
        """Initializer."""
        self.ui_cfg = ui_cfg
        self.display_cfg = display_cfg

        # animation
        self._label_colors: Dict[str, NDArrayF64] = dict()

        figsize = (
            int(self.ui_cfg.width * self.ui_cfg.scale // self.ui_cfg.dpi),
            int(self.ui_cfg.height * self.ui_cfg.scale // self.ui_cfg.dpi),
        )
        self.fig = plt.figure(figsize=figsize, dpi=self.ui_cfg.dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        self.ax.axis("off")

    def write(self, out_path: str) -> None:
        """Write image."""
        self.fig.savefig(out_path, self.ui_cfg.dpi)

    def draw_image(self, name: str, img: NDArrayU8) -> None:
        """Draw image."""
        self.fig.canvas.manager.set_window_title(name)
        self.ax.imshow(img, interpolation="nearest", aspect="auto")

    def _get_label_color(self, label_id: str) -> NDArrayF64:
        """Get color by id (if not found, then create a random color)."""
        if label_id not in self._label_colors:
            self._label_colors[label_id] = random_color()
        return self._label_colors[label_id]

    def show_frame_attributes(self, frame: Frame) -> None:
        """Visualize attribute infomation of a frame."""
        if frame.attributes is None or len(frame.attributes) == 0:
            return
        attributes = frame.attributes
        key_width = 0
        for k, _ in attributes.items():
            if len(k) > key_width:
                key_width = len(k)
        attr_tag = io.StringIO()
        for k, v in attributes.items():
            attr_tag.write("{}: {}\n".format(k.rjust(key_width, " "), v))
        attr_tag.seek(0)
        self.ax.text(
            25,
            90,
            attr_tag.read()[:-1],
            fontproperties=self.ui_cfg.font,
            color="red",
            bbox={"facecolor": "white", "alpha": 0.4, "pad": 10, "lw": 0},
        )

    def _draw_label_attributes(
        self, label: Label, x_coord: float, y_coord: float
    ) -> None:
        """Visualize attribute infomation of a label."""
        text = (
            label.category
            if label.category is not None
            else self.ui_cfg.default_category
        )
        attributes = label.attributes if label.attributes is not None else {}
        if "occluded" in attributes and attributes["occluded"]:
            text += ",o"
        if "truncated" in attributes and attributes["truncated"]:
            text += ",t"
        if check_crowd(label):
            text += ",c"
        self.ax.text(
            x_coord,
            y_coord,
            text,
            fontsize=int(10 * self.ui_cfg.scale),
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.5,
                "boxstyle": "square,pad=0.1",
            },
        )

    def _draw_box(self, labels: List[Label], gen_box_func: GenBoxFunc) -> None:
        """Draw Box on the axes."""
        for label in labels:
            occluded = False
            if label.attributes is not None and "occluded" in label.attributes:
                occluded = bool(label.attributes["occluded"])
            color = self._get_label_color(label.id).tolist()
            alpha = 0.5 if occluded else 0.8
            for result in gen_box_func(label, color, occluded, alpha):
                self.ax.add_patch(result)

            if self.display_cfg.show_tags:
                self._draw_label_attributes(
                    label,
                    label.box2d.x1,  # type: ignore
                    (label.box2d.y1 - 4),  # type: ignore
                )

    def draw_box2d(self, labels: List[Label]) -> None:
        """Draw Box2d on the axes."""
        labels = [label for label in labels if label.box2d is not None]
        self._draw_box(labels, gen_2d_rect)

    def draw_box3d(self, labels: List[Label], intrinsics) -> None:
        """Draw Box3d on the axes."""
        labels = [label for label in labels if label.box3d is not None]
        self._draw_box(labels, partial(gen_3d_cube, intrinsics=intrinsics))

    def draw_poly2d(self, labels: List[Label]) -> None:
        """Draw poly2d labels not in 'lane' and 'drivable' categories."""
        for label in labels:
            if label.poly2d is None:
                continue
            if label.category in ["drivable area", "lane"]:
                continue

            color = self._get_label_color(label.id)
            alpha = 0.5

            # Record the tightest bounding box
            x1, x2 = self.ui_cfg.width * self.ui_cfg.scale, 0.0
            y1, y2 = self.ui_cfg.height * self.ui_cfg.scale, 0.0
            for poly in label.poly2d:
                patch = poly2patch(
                    poly.vertices,
                    poly.types,
                    closed=poly.closed,
                    alpha=alpha,
                    color=color,
                )
                self.ax.add_patch(patch)

                if self.display_cfg.show_ctrl_points:
                    self._draw_ctrl_points(
                        poly.vertices, poly.types, color, alpha
                    )

                patch_vertices = np.array(poly.vertices)
                x1 = min(np.min(patch_vertices[:, 0]), x1)
                y1 = min(np.min(patch_vertices[:, 1]), y1)
                x2 = max(np.max(patch_vertices[:, 0]), x2)
                y2 = max(np.max(patch_vertices[:, 1]), y2)

            # Show attributes
            if self.display_cfg.show_tags:
                self._draw_label_attributes(
                    label,
                    x1 + (x2 - x1) * 0.4,
                    y1 - 3.5,
                )

    def _draw_ctrl_points(
        self,
        vertices: List[Tuple[float, float]],
        types: str,
        color: NDArrayF64,
        alpha: float,
    ) -> None:
        """Draw the polygon vertices / control points."""
        for idx, vert_data in enumerate(zip(vertices, types)):
            vert = vert_data[0]
            vert_type = vert_data[1]

            # Add the point first
            self.ax.add_patch(
                mpatches.Circle(
                    vert,
                    self.display_cfg.ctrl_point_size,
                    alpha=alpha,
                    color=color,
                )
            )
            # Draw the dashed line to the previous vertex.
            if vert_type == "C":
                if idx == 0:
                    vert_prev = vertices[-1]
                else:
                    vert_prev = vertices[idx - 1]
                edge = np.concatenate(
                    [
                        np.array(vert_prev)[None, ...],
                        np.array(vert)[None, ...],
                    ],
                    axis=0,
                )
                self.ax.add_patch(
                    mpatches.Polygon(
                        edge,
                        linewidth=int(2 * self.ui_cfg.scale),
                        linestyle=(1, (1, 2)),
                        edgecolor=color,
                        facecolor="none",
                        fill=False,
                        alpha=alpha,
                    )
                )

                # Draw the dashed line to the next vertex.
                if idx == len(vertices) - 1:
                    vert_next = vertices[0]
                    type_next = types[0]
                else:
                    vert_next = vertices[idx + 1]
                    type_next = types[idx + 1]

                if type_next == "L":
                    edge = np.concatenate(
                        [
                            np.array(vert_next)[None, ...],
                            np.array(vert)[None, ...],
                        ],
                        axis=0,
                    )
                    self.ax.add_patch(
                        mpatches.Polygon(
                            edge,
                            linewidth=int(2 * self.ui_cfg.scale),
                            linestyle=(1, (1, 2)),
                            edgecolor=color,
                            facecolor="none",
                            fill=False,
                            alpha=alpha,
                        )
                    )
