"""
An offline label visualizer for Scalable file.
Works for 2D / 3D bounding box, segmentation masks, etc.
"""

import argparse
import io
import os
import urllib.request
from typing import Any, Dict, List
from threading import Timer

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

from scalabel.label.typing import Frame, Box2D, Box3D, Intrinsics
from scalabel.label.io import load
from scalabel.vis.helper import get_intrinsic_matrix, random_color
from scalabel.vis.geometry import Label3d


class LabelViewer:
    """Visualize 2D and 3D bounding boxes.

    Keymap:
    -  N / P: Show next or previous image
    -  Space: Start / stop animation
    -  T: Toggle 2D / 3D bounding box (if avaliable)
    -  Y: Toggle image / segmentation view (if avaliable)

    Export images:
    - add `-o {dir}` tag when runing
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, args) -> None:
        """initializer"""
        self.ax = None
        self.fig = None
        self.frame_index: int = 0
        self.start_index: int = 0
        self.scale: float = args.scale

        self.image_dir: str = args.image_dir
        self.out_dir: str = args.output_dir

        if os.path.exists(args.labels):
            self.frames: List[Frame] = load(args.labels, use_obj_prarser=True)
        else:
            self.frames: List[Frame] = load(
                os.path.join(args.image_dir, args.labels), use_obj_prarser=True
            )
        print("Load images: ", len(self.frames))

        # parameters for UI
        self.with_attr: bool = True
        self.with_box2d: bool = False
        self.with_box3d: bool = True
        self.show_seg: bool = False
        self.image_width: int = args.width
        self.image_height: int = args.height
        self.default_category: str = "Car"

        # Matplotlib font
        self.font = FontProperties()
        self.font.set_family(["Aerial", "monospace"])
        self.font.set_weight("bold")
        self.font.set_size(18 * self.scale)

        self.is_running: bool = False
        self.interval: float = 0.4
        self._timer: Timer = Timer(self.interval, self.tick)

        self._label_colors: Dict[str, Any] = dict()

    def view(self) -> None:
        """start the visualization"""
        self.frame_index = 0
        if self.out_dir is None:
            self.init_show_window()
        else:
            self.write()

    def init_show_window(self, width=16, height=9, dpi=100):
        """read and draw image"""
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

        plt.connect("key_release_event", self.key_press)
        self.show_frame()
        plt.show()

    def key_press(self, event):
        """handel control keys"""
        if event.key == "n":
            self.frame_index += 1
        elif event.key == "p":
            self.frame_index -= 1
        elif event.key == "t":
            self.with_box2d = not self.with_box2d
            self.with_box3d = not self.with_box3d
        elif event.key == "y":
            self.show_seg = not self.show_seg
        elif event.key == " ":
            if not self.is_running:
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

    def tick(self) -> bool:
        """animation tick"""
        self.is_running = False
        self.start_animation()
        self.frame_index += 1
        self.show_frame()
        plt.draw()

    def start_animation(self) -> bool:
        """start the animation timer"""
        if not self.is_running:
            self._timer.start()
            self.is_running = True

    def stop_animation(self) -> bool:
        """stop the animation timer"""
        self._timer.cancel()
        self.is_running = False

    def show_frame(self) -> bool:
        """show one frame in matplotlib axes"""
        plt.cla()

        frame = self.frames[self.frame_index % len(self.frames)]
        print("Image:", frame.name)
        self.fig.canvas.set_window_title(frame.name)

        # show image
        if frame.url is not None and len(frame.url) > 0:
            image_data = urllib.request.urlopen(frame.url, timeout=300).read()
            im = np.asarray(Image.open(io.BytesIO(image_data)))
        else:
            image_path = os.path.join(self.image_dir, frame.name)
            print("Local path:", image_path)
            img = Image.open(image_path)
            im = np.array(img, dtype=np.uint8)

        if self.show_seg:
            image_seg_path = os.path.join(
                self.image_dir, frame.name.replace("img", "seg")
            )
            if os.path.exists(image_seg_path):
                print("Local segmentation image path:", image_seg_path)
                img_seg = Image.open(image_seg_path)
                im_seg = np.array(img_seg, dtype=np.uint8)

                self.ax.imshow(im_seg, interpolation="nearest", aspect="auto")
                return True

        self.ax.imshow(im, interpolation="nearest", aspect="auto")

        # show label
        if frame.labels is None or len(frame.labels) == 0:
            print("No labels found")
            return True

        labels = frame.labels
        # print(labels)

        if self.with_attr:
            self.show_frame_attributes(frame)

        if self.with_box2d:
            for b in labels:
                attributes = {}
                if b.attributes is not None:
                    attributes = b.attributes
                if b.box_2d is not None:
                    self.ax.add_patch(self.gen_2d_rect(b.id, b.box_2d))
                    text = (
                        b.category
                        if b.category is not None
                        else self.default_category
                    )
                    if "occluded" in attributes and attributes["occluded"]:
                        text += ",o"
                    if "truncated" in attributes and attributes["truncated"]:
                        text += ",t"
                    if "crowd" in attributes and attributes["crowd"]:
                        text += ",c"
                    self.ax.text(
                        (b.box_2d.x1) * self.scale,
                        (b.box_2d.y1 - 4) * self.scale,
                        text,
                        fontsize=10 * self.scale,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.5,
                            "boxstyle": "square,pad=0.1",
                        },
                    )

        if self.with_box3d:
            for b in labels:
                attributes = {}
                if b.attributes is not None:
                    attributes = b.attributes
                if b.box_3d is not None and frame.intrinsics is not None:
                    occluded = False
                    if "occluded" in attributes:
                        occluded = attributes["occluded"]

                    for line in self.gen_3d_cube(
                        b.id, b.box_3d, frame.intrinsics, occluded
                    ):
                        self.ax.add_patch(line)

                    text = (
                        b.category
                        if b.category is not None
                        else self.default_category
                    )

                    self.ax.text(
                        (b.box_2d.x1) * self.scale,
                        (b.box_2d.y1 - 4) * self.scale,
                        text,
                        fontsize=10 * self.scale,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.5,
                            "boxstyle": "square,pad=0.1",
                        },
                    )

        self.ax.axis("off")
        return True

    def get_label_color(self, label_id) -> Dict[str, Any]:
        """get color by id (if not found, then create a random color)"""
        if label_id not in self._label_colors:
            self._label_colors[label_id] = random_color()
        return self._label_colors[label_id]

    def gen_2d_rect(self, label_id: str, box2d: Box2D):
        """generate individual bounding box from label."""
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
            linewidth=2 * self.scale,
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
    ):
        """generate individual bounding box from 3d label."""
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
                    linewidth=2 * self.scale,
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
                    linewidth=2 * self.scale,
                    edgecolor=box_color,
                    facecolor="none",
                    fill=False,
                    alpha=alpha,
                )
            )

        return lines

    def write(self, width=16, height=9, dpi=100) -> bool:
        """save visualized result to file"""
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
            out_path = os.path.join(self.out_dir, out_name)
            if self.show_frame():
                self.fig.savefig(out_path, dpi=dpi)
                out_paths.append(out_path)

            self.frame_index += 1
            if self.frame_index >= len(self.frames):
                self.start_index = self.frame_index
        return True

    def show_frame_attributes(self, frame) -> bool:
        """visualize attribute infomation of a frame"""
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
            25 * self.scale,
            90 * self.scale,
            attr_tag.read()[:-1],
            fontproperties=self.font,
            color="red",
            bbox={"facecolor": "white", "alpha": 0.4, "pad": 10, "lw": 0},
        )
        return True


def parse_args():
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
        "--no-lane",
        action="store_true",
        default=False,
        help="Do not show lanes",
    )
    parser.add_argument(
        "--no-drivable",
        action="store_true",
        default=False,
        help="Do not show drivable areas",
    )
    parser.add_argument(
        "--no-box2d",
        action="store_true",
        default=False,
        help="Do not show 2D bounding boxes",
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
    parser.add_argument(
        "--target-objects",
        type=str,
        default="",
        help="A comma separated list of objects. If this is "
        "not empty, only show images with the target "
        "objects.",
    )
    args = parser.parse_args()

    if len(args.target_objects) > 0:
        args.target_objects = args.target_objects.split(",")

    return args


def main():
    """main function"""
    args = parse_args()
    viewer = LabelViewer(args)
    viewer.view()


if __name__ == "__main__":
    main()
