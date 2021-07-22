"""An offline visualzation controller for Scalabel file."""

import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from threading import Timer
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt

from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import NDArrayF64, NDArrayU8
from ..label.io import load
from .helper import fetch_image
from .viewer import DisplayConfig, LabelViewer, UIConfig


@dataclass
class ControllerConfig:
    """Visulizer's config class."""

    # path
    image_dir: str
    out_dir: str

    # content
    with_attr: bool
    with_box2d: bool
    with_box3d: bool
    with_poly2d: bool

    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        with_attr: bool = True,
        with_box2d: bool = True,
        with_box3d: bool = False,
        with_poly2d: bool = True,
    ) -> None:
        """Initialize with args."""
        self.image_dir = image_dir
        self.out_dir = output_dir
        self.with_attr = with_attr
        self.with_box2d = with_box2d
        self.with_box3d = with_box3d
        self.with_poly2d = with_poly2d


class ViewController:
    """Visualization controller for Scalabel.

    Keymap:
    -  n / p: Show next or previous image
    -  Space: Start / stop animation
    -  t: Toggle 2D / 3D bounding box (if avaliable)
    -  y: Toggle image / segmentation view (if avaliable)
    -  a: Toggle the display of the attribute tags on boxes or polygons.
    -  c: Toggle the display of polygon vertices.
    -  Up: Increase the size of polygon vertices.
    -  Down: Decrease the size of polygon vertices.

    Export images:
    - add `-o {dir}` tag when runing.
    """

    def __init__(
        self,
        config: ControllerConfig,
        viewer: LabelViewer,
        inp_path: str,
        nproc: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        """Initializer."""
        self.config = config
        self.viewer = viewer

        self.frame_index: int = 0

        # animation
        self._run_animation: bool = False
        self._timer: Timer = Timer(0.4, self.tick)
        self._label_colors: Dict[str, NDArrayF64] = dict()

        # load label file
        print("Label file:", inp_path)
        if not os.path.exists(inp_path):
            logger.error("Label file not found!")
        self.frames = load(inp_path, nproc).frames
        logger.info("Load images: %d", len(self.frames))

        self.images: Dict[str, "concurrent.futures.Future[NDArrayU8]"] = dict()
        # Cache the images in separate threads.
        for frame in self.frames:
            self.images[frame.name] = executor.submit(
                fetch_image, (frame, self.config.image_dir)
            )

    def view(self) -> None:
        """Start the visualization."""
        self.frame_index = 0
        if self.config.out_dir is None:
            plt.connect("key_release_event", self.key_press)
            self.show_frame()
            self.viewer.show()
        else:
            os.makedirs(self.config.out_dir, exist_ok=True)
            while self.frame_index < len(self.frames):
                out_name = (
                    os.path.splitext(
                        os.path.split(self.frames[self.frame_index].name)[1]
                    )[0]
                    + ".png"
                )
                out_path = os.path.join(self.config.out_dir, out_name)
                logger.info("Writing %s", out_path)
                self.show_frame()
                self.viewer.write(out_path)
                self.frame_index += 1

    def key_press(self, event: matplotlib.backend_bases.Event) -> None:
        """Handel control keys."""
        if event.key == "n":
            self.frame_index += 1
        elif event.key == "p":
            self.frame_index -= 1
        elif event.key == "t":
            self.config.with_box2d = not self.config.with_box2d
            self.config.with_box3d = not self.config.with_box3d
        elif event.key == " ":
            if not self._run_animation:
                self.start_animation()
            else:
                self.stop_animation()
        elif event.key == "a":
            self.viewer.display_cfg.show_tags = (
                not self.viewer.display_cfg.show_tags
            )
        # Control keys for polygon mode
        elif event.key == "c":
            self.viewer.display_cfg.show_ctrl_points = (
                not self.viewer.display_cfg.show_ctrl_points
            )
        elif event.key == "up":
            self.viewer.display_cfg.ctrl_point_size = min(
                15.0, self.viewer.display_cfg.ctrl_point_size + 0.5
            )
        elif event.key == "down":
            self.viewer.display_cfg.ctrl_point_size = max(
                0.0, self.viewer.display_cfg.ctrl_point_size - 0.5
            )
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
        # Fetch the image
        img = self.images[frame.name].result()
        self.viewer.draw_image(img, frame.name)

        # show label
        if frame.labels is None or len(frame.labels) == 0:
            print("No labels found")
            return True

        labels = frame.labels
        if self.config.with_attr:
            self.viewer.draw_attributes(frame)
        if self.config.with_box2d:
            self.viewer.draw_box2ds(labels)
        if self.config.with_box3d and frame.intrinsics is not None:
            self.viewer.draw_box3ds(labels, frame.intrinsics)
        if self.config.with_poly2d:
            self.viewer.draw_poly2ds(labels)

        return True


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
        "-s",
        "--scale",
        type=float,
        default=1.0,
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
        default=720,
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
        "--output_dir",
        required=False,
        default=None,
        type=str,
        help="output image directory with label visualization. "
        "If it is set, the images will be written to the "
        "output folder instead of being displayed "
        "interactively.",
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
            show_ctrl_points=not args.no_vertices,
            show_tags=not args.no_tags,
            ctrl_points_size=2.0,
        )
        ctrl_cfg = ControllerConfig(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
        )
        viewer = LabelViewer(ui_cfg, display_cfg)
        controller = ViewController(
            ctrl_cfg, viewer, args.labels, args.nproc, executor
        )
        controller.view()


if __name__ == "__main__":
    main()
