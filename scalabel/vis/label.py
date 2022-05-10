"""An offline label visualizer for Scalable file.

Works for 2D / 3D bounding box, segmentation masks, etc.
"""

import argparse
import concurrent.futures
import io
import threading
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from skimage.transform import resize

from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayF64, NDArrayU8
from ..label.transforms import rle_to_mask
from ..label.typing import Config, Edge, Frame, Intrinsics, Label, Node
from ..label.utils import (
    check_crowd,
    check_ignored,
    check_occluded,
    check_truncated,
    get_leaf_categories,
)
from .controller import (
    ControllerConfig,
    DisplayConfig,
    DisplayData,
    ViewController,
)
from .helper import (
    gen_2d_rect,
    gen_3d_cube,
    gen_graph_edge,
    gen_graph_point,
    poly2patch,
    random_color,
)

# Necessary due to Queue being generic in stubs but not at runtime
# https://mypy.readthedocs.io/en/stable/runtime_troubles.html#not-generic-runtime
if TYPE_CHECKING:
    DisplayDataQueue = Queue[  # pylint: disable=unsubscriptable-object
        DisplayData
    ]
else:
    DisplayDataQueue = Queue


@dataclass
class UIConfig:
    """Visualizer UI's config class."""

    height: int
    width: int
    scale: float
    dpi: int = 80
    font: FontProperties = FontProperties(
        family=["sans-serif", "monospace"], weight="bold", size=18
    )
    default_category: str = "car"

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        height: int = 720,
        width: int = 1280,
        scale: float = 1.0,
        dpi: int = 80,
        weight: str = "bold",
        family: List[str] = ["sans-serif", "monospace"],
    ) -> None:
        """Initialize with default values."""
        self.height = height
        self.width = width
        self.scale = scale
        self.dpi = dpi
        self.font.set_size(int(18 * scale))
        self.font.set_weight(weight)
        self.font.set_family(family)


def _get_node_color(node: Node, graph_type: Optional[str] = "") -> List[float]:
    """Get node color based on visibility."""
    if not graph_type:
        return [0.0, 0.0, 0.0]
    if graph_type.startswith("Pose2D"):
        if graph_type.endswith("Pred"):
            return [1.0, 1.0, 0.0]
        if node.visibility == "V":  # visible keypoints
            color = [1.0, 1.0, 0.0]
        elif node.visibility == "N":  # not visible keypoints
            color = [1.0, 0.0, 0.0]
        else:  # cannot be estimated keypoints
            color = [0.0, 0.0, 0.0]
        return color
    return [1.0, 1.0, 0.0]


def _get_edge_color(edge: Edge, graph_type: Optional[str] = "") -> List[float]:
    """Get edge color based on visibility."""
    if not graph_type:
        return [0.0, 0.0, 0.0]
    if graph_type.startswith("Pose2D"):
        if edge.type == "body":
            color = [0.0, 1.0, 0.0]
        elif edge.type == "left_side":
            color = [0.0, 0.0, 1.0]
        else:
            color = [1.0, 0.0, 0.0]
        return color
    return [0.0, 0.0, 0.0]


class LabelViewer:
    """Visualize 2D, 3D bounding boxes and 2D polygons.

    The class provides both high-level APIs and middle-level APIs to visualize
    the image and the corresponding Scalabel-format labels.

    High-level APIs:
        draw(image: np.array, frame: scalabel.label.typing.Frame):
            Draw the image with the labels stored in the 'frame'.
        show(): display the visualization of the current image.
        save(out_path: str): save the visualization of the current image.

    Middle-level APIs:
        draw_image(image: np.array): plot the current image
        draw_attributes(frame: Frame): plot the frame attributes
        draw_box2ds(labels: list[Label]): plot 2D bounding boxes
        draw_box3ds(labels: list[Label], intrinsics: Intrisics):
            plot 3D bounding boxes
        draw_poly2ds(labels: list[Label], alpha: float):
            plot 2D polygons with the given alpha value
        draw_graph(labels: list[Label]): plot graph
    """

    def __init__(
        self,
        ui_cfg: UIConfig = UIConfig(),
        label_cfg: Optional[Config] = None,
    ) -> None:
        """Initialize the label viewer."""
        self.ui_cfg = ui_cfg

        # if specified, use category colors in config
        self._category_colors: Optional[Dict[str, NDArrayF64]] = None
        if label_cfg:
            self._category_colors = {
                c.name: (np.asarray(c.color) / 255)
                if c.color
                else random_color()
                for c in get_leaf_categories(label_cfg.categories)
            }

        # animation
        self._label_colors: Dict[str, NDArrayF64] = {}

        figsize = (
            int(self.ui_cfg.width * self.ui_cfg.scale // self.ui_cfg.dpi),
            int(self.ui_cfg.height * self.ui_cfg.scale // self.ui_cfg.dpi),
        )
        self.fig = plt.figure(figsize=figsize, dpi=self.ui_cfg.dpi)
        self.ax: Axes = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        self.ax.axis("off")

    def run_with_controller(self, controller: ViewController) -> None:
        """Start running with the controller."""
        threading.Thread(target=self._worker, args=(controller.queue,)).start()
        controller.run()

    def _worker(self, queue: DisplayDataQueue) -> None:
        """Worker to collaborate with the controller."""
        while True:
            data: DisplayData = queue.get()
            self.draw(data.image, data.frame)
            if data.out_path:
                self.save(data.out_path)
            else:
                plt.draw()

    def show(self) -> None:  # pylint: disable=no-self-use
        """Show the visualization."""
        plt.show()

    def save(self, out_path: str) -> None:  # pylint: disable=no-self-use
        """Save the visualization."""
        plt.savefig(out_path, dpi=self.ui_cfg.dpi)

    def draw(  # pylint: disable=too-many-arguments
        self,
        image: NDArrayU8,
        frame: Frame,
        with_attr: bool = True,
        with_box2d: bool = True,
        with_box3d: bool = True,
        with_poly2d: bool = True,
        with_graph: bool = True,
        with_rle: bool = True,
        with_ctrl_points: bool = False,
        with_tags: bool = True,
        ctrl_point_size: float = 2.0,
        alpha: float = 0.5,
    ) -> None:
        """Display the image and corresponding labels."""
        plt.cla()
        img = resize(image, (self.ui_cfg.height, self.ui_cfg.width))
        self.draw_image(img, frame.name)
        if frame.labels is None or len(frame.labels) == 0:
            logger.info("No labels found")
            return

        # If both poly2d and rle are specified, show only rle if labels have
        # rle and otherwise poly2d.
        has_rle = any(label.rle is not None for label in frame.labels)

        labels = frame.labels
        if with_attr:
            self.draw_attributes(frame)
        if with_box2d:
            self.draw_box2ds(labels, with_tags=with_tags)
        if with_box3d and frame.intrinsics is not None:
            self.draw_box3ds(labels, frame.intrinsics, with_tags=with_tags)
        if with_poly2d and (not with_rle or not has_rle):
            self.draw_poly2ds(
                labels,
                with_tags=with_tags,
                with_ctrl_points=with_ctrl_points,
                ctrl_point_size=ctrl_point_size,
                alpha=alpha,
            )
        if with_graph:
            self.draw_graph(labels, alpha=alpha)
        if with_rle and has_rle:
            self.draw_rle(img, labels, alpha=alpha)

    def draw_image(self, img: NDArrayU8, title: Optional[str] = None) -> None:
        """Draw image."""
        if title is not None:
            self.fig.canvas.manager.set_window_title(title)
        self.ax.imshow(img, interpolation="bilinear", aspect="auto")

    def _get_label_color(self, label: Label) -> NDArrayF64:
        """Get color by id (if not found, then create a random color)."""
        category = label.category
        if (
            self._category_colors
            and category
            and category in self._category_colors
        ):
            return self._category_colors[category]

        label_id = label.id
        if label_id not in self._label_colors:
            self._label_colors[label_id] = random_color()
        return self._label_colors[label_id]

    def draw_attributes(self, frame: Frame) -> None:
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
            attr_tag.write(f"{k.rjust(key_width, ' ')}: {v}\n")
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
        if check_truncated(label):
            text += ",t"
        if check_occluded(label):
            text += ",o"
        if check_crowd(label):
            text += ",c"
        if check_ignored(label):
            text += ",i"
        if label.score is not None:
            text += f"{label.score:.2f}"
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

    def draw_box2ds(self, labels: List[Label], with_tags: bool = True) -> None:
        """Draw Box2d on the axes."""
        for label in labels:
            if label.box2d is not None:
                color = self._get_label_color(label).tolist()
                for result in gen_2d_rect(
                    label, color, int(2 * self.ui_cfg.scale)
                ):
                    self.ax.add_patch(result)

                if with_tags:
                    self._draw_label_attributes(
                        label, label.box2d.x1, (label.box2d.y1 - 4)
                    )

    def draw_box3ds(
        self,
        labels: List[Label],
        intrinsics: Intrinsics,
        with_tags: bool = True,
    ) -> None:
        """Draw Box3d on the axes."""
        for label in labels:
            if label.box3d is not None:
                color = self._get_label_color(label).tolist()
                occluded = check_occluded(label)
                alpha = 0.5 if occluded else 0.8
                for result in gen_3d_cube(
                    label, color, int(2 * self.ui_cfg.scale), intrinsics, alpha
                ):
                    self.ax.add_patch(result)

                if with_tags and label.box2d is not None:
                    self._draw_label_attributes(
                        label, label.box2d.x1, (label.box2d.y1 - 4)
                    )

    def draw_poly2ds(
        self,
        labels: List[Label],
        alpha: float = 0.5,
        with_tags: bool = True,
        with_ctrl_points: bool = False,
        ctrl_point_size: float = 2.0,
    ) -> None:
        """Draw poly2d labels not in 'lane' and 'drivable' categories."""
        for label in labels:
            if label.poly2d is None:
                continue
            color = self._get_label_color(label)

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

                if with_ctrl_points:
                    self._draw_ctrl_points(
                        poly.vertices,
                        poly.types,
                        color,
                        alpha,
                        ctrl_point_size,
                    )

                patch_vertices: NDArrayF64 = np.array(
                    poly.vertices, dtype=np.float64
                )
                x1 = min(np.min(patch_vertices[:, 0]), x1)
                y1 = min(np.min(patch_vertices[:, 1]), y1)
                x2 = max(np.max(patch_vertices[:, 0]), x2)
                y2 = max(np.max(patch_vertices[:, 1]), y2)

            # Show attributes
            if with_tags:
                self._draw_label_attributes(
                    label, x1 + (x2 - x1) * 0.4, y1 - 3.5
                )

    def _draw_ctrl_points(
        self,
        vertices: List[Tuple[float, float]],
        types: str,
        color: NDArrayF64,
        alpha: float,
        ctrl_point_size: float = 2.0,
    ) -> None:
        """Draw the polygon vertices / control points."""
        for idx, vert_data in enumerate(zip(vertices, types)):
            vert = vert_data[0]
            vert_type = vert_data[1]

            # Add the point first
            self.ax.add_patch(
                mpatches.Circle(
                    vert, ctrl_point_size, alpha=alpha, color=color
                )
            )
            # Draw the dashed line to the previous vertex.
            if vert_type == "C":
                if idx == 0:
                    vert_prev = vertices[-1]
                else:
                    vert_prev = vertices[idx - 1]
                edge: NDArrayF64 = np.concatenate(
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

    def draw_graph(
        self,
        labels: List[Label],
        linewidth: int = 2,
        pointsize: int = 2,
        alpha: float = 0.5,
    ) -> None:
        """Draw Graph on the axes."""
        for label in labels:
            if label.graph is not None:
                for edge in label.graph.edges:
                    color = _get_edge_color(edge, label.graph.type)
                    width = int(linewidth * self.ui_cfg.scale)
                    result = gen_graph_edge(edge, label, color, width, alpha)
                    self.ax.add_patch(result[0])
                for node in label.graph.nodes:
                    color = _get_node_color(node, label.graph.type)
                    result = gen_graph_point(
                        node, color, int(pointsize * self.ui_cfg.scale), alpha
                    )
                    self.ax.add_patch(result[0])

    def draw_rle(
        self, image: NDArrayU8, labels: List[Label], alpha: float = 0.5
    ) -> None:
        """Draw RLE."""
        combined_mask: NDArrayU8 = np.zeros(image.shape, dtype=np.uint8)

        labels = sorted(labels, key=lambda label: float(label.score or 0))

        for label in labels:
            if not label.rle:
                continue

            color: NDArrayF64 = self._get_label_color(label) * 255
            bitmask = resize(
                rle_to_mask(label.rle),
                (self.ui_cfg.height, self.ui_cfg.width),
            )
            mask = np.repeat(bitmask[:, :, np.newaxis], 3, axis=2)

            # Non-zero values correspond to colors for each label
            combined_mask = np.where(
                mask, color.astype(np.uint8), combined_mask
            )

        img: NDArrayU8 = image * 255
        self.ax.imshow(
            np.where(
                combined_mask > 0,
                combined_mask.astype(np.uint8),
                img.astype(np.uint8),
            ),
            alpha=alpha,
        )


def parse_args() -> argparse.Namespace:
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser(
        """
Interface keymap:
    -  n / p: Show next or previous image
    -  Space: Start / stop animation
    -  t: Toggle 2D / 3D bounding box (if avaliable)
    -  a: Toggle the display of the attribute tags on boxes or polygons.
    -  c: Toggle the display of polygon vertices.
    -  Up: Increase the size of polygon vertices.
    -  Down: Decrease the size of polygon vertices.

Export images:
    - add `-o {dir}` tag when runing.
    """
    )
    parser.add_argument("-i", "--image-dir", help="image directory")
    parser.add_argument(
        "-l",
        "--labels",
        required=False,
        default="labels.json",
        help="Path to the json file",
        type=str,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the image (px)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the image (px)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=1.0,
        help="Scale up factor for annotation factor. "
        "Useful when producing visualization as thumbnails.",
    )
    parser.add_argument(
        "--no-attr",
        action="store_true",
        default=False,
        help="Do not show attributes",
    )
    parser.add_argument(
        "--no-box3d",
        action="store_true",
        default=True,
        help="Do not show 3D bounding boxes",
    )
    parser.add_argument(
        "--no-tags",
        action="store_true",
        default=False,
        help="Do not show tags on boxes or polygons",
    )
    parser.add_argument(
        "--no-vertices",
        action="store_true",
        default=False,
        help="Do not show vertices",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=None,
        type=str,
        help="output image directory with label visualization. "
        "If it is set, the images will be written to the "
        "output folder instead of being displayed "
        "interactively.",
    )
    parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="from which frame to visualize. Default is 0.",
    )
    parser.add_argument(
        "--range-end",
        type=int,
        default=-1,
        help="up to which frame to visualize. Default is -1, "
        "indicating loading all frames for visualizatoin.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for json loading",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()
    # Initialize the thread executor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        ui_cfg = UIConfig(
            height=args.height,
            width=args.width,
            scale=args.scale,
        )
        display_cfg = DisplayConfig(
            with_attr=not args.no_attr,
            with_box2d=args.no_box3d,
            with_box3d=not args.no_box3d,
            with_ctrl_points=not args.no_vertices,
            with_tags=not args.no_tags,
        )
        viewer = LabelViewer(ui_cfg)

        ctrl_cfg = ControllerConfig(
            image_dir=args.image_dir,
            label_path=args.labels,
            out_dir=args.output_dir,
            nproc=args.nproc,
            range_begin=args.range_begin,
            range_end=args.range_end,
        )
        controller = ViewController(ctrl_cfg, display_cfg, executor)
        viewer.run_with_controller(controller)


if __name__ == "__main__":
    main()
