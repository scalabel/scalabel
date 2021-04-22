"""An offline label visualizer for Scalable file.

Works for 2D / 3D bounding box, segmentation masks, etc.
"""

import argparse
import io
import os
import sys
import urllib.request
from dataclasses import dataclass
from threading import Timer
from typing import Dict, List

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image

from ..label.io import load
from ..label.typing import Box2D, Box3D, Frame, Intrinsics
from .geometry import Label3d
from .helper import get_intrinsic_matrix, random_color


@dataclass
class ViewerConfig:
    """Visulizer's config class."""

    # path
    image_dir: str
    out_dir: str
    scale: float

    # content
    show_seg: bool = False
    with_attr: bool = True
    with_box2d: bool = True
    with_box3d: bool = False

    # parameters for UI
    image_width: int = 1280
    image_height: int = 800
    default_category: str = "Car"
    font = FontProperties()  # for Matplotlib font

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize with args."""
        self.image_dir = args.image_dir
        self.out_dir = args.output_dir
        self.scale = args.scale
        self.image_width = args.width
        self.image_height = args.height
        self.with_attr = not args.no_attr


class LabelViewer:
    """Visualize 2D and 3D bounding boxes.

    Keymap:
    -  N / P: Show next or previous image
    -  Space: Start / stop animation
    -  T: Toggle 2D / 3D bounding box (if avaliable)
    -  Y: Toggle image / segmentation view (if avaliable)

    Export images:
    - add `-o {dir}` tag when runing.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializer."""
        self.config = ViewerConfig(args)

        self.ax: matplotlib.axes.Axes = None
        self.fig: matplotlib.figure.Figure = None
        self.frame_index: int = 0
        self.start_index: int = 0
        self.no_box3d = args.no_box3d

        # set fonts
        self.config.font.set_family(["sans-serif", "monospace"])
        self.config.font.set_weight("bold")
        self.config.font.set_size(18 * self.config.scale)

        # animation
        self._interval: float = 0.4
        self._run_animation: bool = False
        self._timer: Timer = Timer(self._interval, self.tick)
        self._label_colors: Dict[str, np.ndarray] = dict()

        # load label file
        print("Label file:", args.labels)
        self.frames: List[Frame] = []
        if os.path.exists(args.labels):
            self.frames = load(args.labels)
        elif os.path.exists(os.path.join(args.image_dir, args.labels)):
            self.frames = load(os.path.join(args.image_dir, args.labels))
        else:
            print("Label file not found!")
            sys.exit(1)

        print("Load images: ", len(self.frames))

    def view(self) -> None:
        """Start the visualization."""
        self.frame_index = 0
        if self.config.out_dir is None:
            self.init_show_window()
        else:
            self.write()

    def init_show_window(
        self, width: int = 16, height: int = 9, dpi: int = 100
    ) -> None:
        """Read and draw image."""
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

        plt.connect("key_release_event", self.key_press)
        self.show_frame()
        plt.show()

    def key_press(self, event: matplotlib.backend_bases.Event) -> None:
        """Handel control keys."""
        if event.key == "n":
            self.frame_index += 1
        elif event.key == "p":
            self.frame_index -= 1
        elif event.key == "t" and not self.no_box3d:
            self.config.with_box2d = not self.config.with_box2d
            self.config.with_box3d = not self.config.with_box3d
        elif event.key == "y":
            self.config.show_seg = not self.config.show_seg
        elif event.key == " ":
            if not self._run_animation:
                self.start_animation()
            else:
                self.stop_animation()
        else:
            return

        self.frame_index = max(self.frame_index, 0)

        if self.show_frame():
            plt.draw()
        else:
            self.key_press(event)

    def tick(self) -> None:
        """Animation tick."""
        self._run_animation = False
        self.start_animation()
        self.frame_index += 1
        self.show_frame()
        plt.draw()

    def start_animation(self) -> bool:
        """Start the animation timer."""
        if not self._run_animation:
            self._timer.start()
            self._run_animation = True
        return True

    def stop_animation(self) -> bool:
        """Stop the animation timer."""
        self._timer.cancel()
        self._run_animation = False
        return True

    def show_frame(self) -> bool:
        """Show one frame in matplotlib axes."""
        plt.cla()

        frame = self.frames[self.frame_index % len(self.frames)]
        print("Image:", frame.name)
        self.fig.canvas.set_window_title(frame.name)

        # show image
        if frame.url is not None and len(frame.url) > 0:
            image_data = urllib.request.urlopen(frame.url, timeout=300).read()
            im = np.asarray(Image.open(io.BytesIO(image_data)))
        else:
            image_path = os.path.join(self.config.image_dir, frame.name)
            print("Local path:", image_path)
            img = Image.open(image_path)
            im = np.array(img, dtype=np.uint8)

        if self.config.show_seg:
            image_seg_path = os.path.join(
                self.config.image_dir, frame.name.replace("img", "seg")
            )
            if os.path.exists(image_seg_path):
                print("Local segmentation image path:", image_seg_path)
                img_seg = Image.open(image_seg_path)
                im_seg = np.array(img_seg, dtype=np.uint8)

                self.ax.imshow(im_seg, interpolation="nearest", aspect="auto")
                return True
            print("Segmentation mask not found.")

        self.ax.imshow(im, interpolation="nearest", aspect="auto")

        # show label
        if frame.labels is None or len(frame.labels) == 0:
            print("No labels found")
            return True

        labels = frame.labels
        # print(labels)

        if self.config.with_attr:
            self.show_frame_attributes(frame)

        if self.config.with_box2d:
            for b in labels:
                attributes = {}
                if b.attributes is not None:
                    attributes = b.attributes
                if b.box_2d is not None:
                    self.ax.add_patch(self.gen_2d_rect(b.id, b.box_2d))
                    text = (
                        b.category
                        if b.category is not None
                        else self.config.default_category
                    )
                    if "occluded" in attributes and attributes["occluded"]:
                        text += ",o"
                    if "truncated" in attributes and attributes["truncated"]:
                        text += ",t"
                    if "crowd" in attributes and attributes["crowd"]:
                        text += ",c"
                    self.ax.text(
                        (b.box_2d.x1) * self.config.scale,
                        (b.box_2d.y1 - 4) * self.config.scale,
                        text,
                        fontsize=10 * self.config.scale,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.5,
                            "boxstyle": "square,pad=0.1",
                        },
                    )

        if self.config.with_box3d:
            for b in labels:
                attributes = {}
                if b.attributes is not None:
                    attributes = b.attributes
                if b.box_3d is not None and frame.intrinsics is not None:
                    occluded = False
                    if "occluded" in attributes:
                        occluded = bool(attributes["occluded"])

                    for line in self.gen_3d_cube(
                        b.id, b.box_3d, frame.intrinsics, occluded
                    ):
                        self.ax.add_patch(line)

                    text = (
                        b.category
                        if b.category is not None
                        else self.config.default_category
                    )
                    if b.box_2d is not None:
                        self.ax.text(
                            (b.box_2d.x1) * self.config.scale,
                            (b.box_2d.y1 - 4) * self.config.scale,
                            text,
                            fontsize=10 * self.config.scale,
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.5,
                                "boxstyle": "square,pad=0.1",
                            },
                        )

        self.ax.axis("off")
        return True

    def get_label_color(self, label_id: str) -> np.ndarray:
        """Get color by id (if not found, then create a random color)."""
        if label_id not in self._label_colors:
            self._label_colors[label_id] = random_color()
        return self._label_colors[label_id]

    def gen_2d_rect(self, label_id: str, box2d: Box2D) -> mpatches.Rectangle:
        """Generate individual bounding box from label."""
        x1 = box2d.x1
        y1 = box2d.y1
        x2 = box2d.x2
        y2 = box2d.y2

        box_color = self.get_label_color(label_id).tolist()
        # Draw and add one box to the figure
        return mpatches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2 * self.config.scale,
            edgecolor=box_color + [0.75],
            facecolor=box_color + [0.25],
            fill=True,
        )

    def gen_3d_cube(
        self,
        label_id: str,
        box3d: Box3D,
        intrinsics: Intrinsics,
        occluded: bool = False,
    ) -> List[mpatches.Polygon]:
        """Generate individual bounding box from 3d label."""
        label = Label3d.from_box3d(box3d)
        edges = label.get_edges_with_visibility(
            get_intrinsic_matrix(intrinsics)
        )

        box_color = self.get_label_color(label_id)
        alpha = 0.5 if occluded else 0.8

        lines = []
        for edge in edges["dashed"]:
            lines.append(
                mpatches.Polygon(
                    edge,
                    linewidth=2 * self.config.scale,
                    linestyle=(0, (2, 2)),
                    edgecolor=box_color,
                    facecolor="none",
                    fill=False,
                    alpha=alpha,
                )
            )
        for edge in edges["solid"]:
            lines.append(
                mpatches.Polygon(
                    edge,
                    linewidth=2 * self.config.scale,
                    edgecolor=box_color,
                    facecolor="none",
                    fill=False,
                    alpha=alpha,
                )
            )

        return lines

    def write(self, width: int = 16, height: int = 9, dpi: int = 100) -> bool:
        """Save visualized result to file."""
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

        out_paths = []

        self.frame_index = self.start_index
        while self.frame_index < len(self.frames):
            out_name = (
                os.path.splitext(
                    os.path.split(self.frames[self.frame_index].name)[1]
                )[0]
                + ".png"
            )
            out_path = os.path.join(self.config.out_dir, out_name)
            if self.show_frame():
                self.fig.savefig(out_path, dpi=dpi)
                out_paths.append(out_path)

            self.frame_index += 1
            if self.frame_index >= len(self.frames):
                self.start_index = self.frame_index
        return True

    def show_frame_attributes(self, frame: Frame) -> bool:
        """Visualize attribute infomation of a frame."""
        if frame.attributes is None or len(frame.attributes) == 0:
            return False
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
            25 * self.config.scale,
            90 * self.config.scale,
            attr_tag.read()[:-1],
            fontproperties=self.config.font,
            color="red",
            bbox={"facecolor": "white", "alpha": 0.4, "pad": 10, "lw": 0},
        )
        return True


def parse_args() -> argparse.Namespace:
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image", required=False, help="input raw image", type=str
    )
    parser.add_argument("--image-dir", help="image directory")
    parser.add_argument(
        "-l",
        "--labels",
        required=False,
        default="labels.json",
        help="corresponding bounding box annotation (json file)",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=1,
        help="Scale up factor for annotation factor. "
        "Useful when producing visualization as "
        "thumbnails.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the image (px)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Height of the image (px)",
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
        default=False,
        help="Do not show 3D bounding boxes",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        default=None,
        type=str,
        help="output image file with bbox visualization. "
        "If it is set, the images will be written to the "
        "output folder instead of being displayed "
        "interactively.",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()
    viewer = LabelViewer(args)
    viewer.view()


if __name__ == "__main__":
    main()
