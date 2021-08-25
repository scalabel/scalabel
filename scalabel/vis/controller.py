"""An offline visualzation controller for Scalabel file."""

import concurrent.futures
import os
from dataclasses import dataclass
from functools import partial
from queue import Queue
from threading import Timer
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event

from ..common.logger import logger
from ..common.parallel import NPROC
from ..common.typing import (  # pylint: disable=unused-import
    NDArrayF64,
    NDArrayU8,
)
from ..label.io import load
from ..label.typing import Frame
from .helper import fetch_image


@dataclass
class DisplayConfig:
    """Visualizer display's config class."""

    with_attr: bool
    with_box2d: bool
    with_box3d: bool
    with_poly2d: bool
    with_ctrl_points: bool
    with_tags: bool
    ctrl_point_size: float

    def __init__(
        self,
        with_attr: bool = True,
        with_box2d: bool = True,
        with_box3d: bool = False,
        with_poly2d: bool = True,
        with_ctrl_points: bool = False,
        with_tags: bool = True,
        ctrl_point_size: float = 2.0,
    ) -> None:
        """Initialize with default values."""
        self.with_attr = with_attr
        self.with_box2d = with_box2d
        self.with_box3d = with_box3d
        self.with_poly2d = with_poly2d
        self.with_ctrl_points = with_ctrl_points
        self.with_tags = with_tags
        self.ctrl_point_size = ctrl_point_size


@dataclass
class DisplayData:
    """The data to be displayed."""

    image: NDArrayU8
    frame: Frame
    display_cfg: DisplayConfig
    out_path: Optional[str]

    def __init__(
        self,
        image: NDArrayU8,
        frame: Frame,
        display_cfg: DisplayConfig,
        out_path: Optional[str] = None,
    ) -> None:
        """Initialize with default values."""
        self.image = image
        self.frame = frame
        self.out_path = out_path
        self.display_cfg = display_cfg


@dataclass
class ControllerConfig:
    """Visulizer's config class."""

    image_dir: str
    label_path: str
    out_dir: str
    nproc: int

    def __init__(
        self, image_dir: str, label_path: str, out_dir: str, nproc: int = NPROC
    ) -> None:
        """Initialize with default values."""
        self.image_dir = image_dir
        self.label_path = label_path
        self.out_dir = out_dir
        self.nproc = nproc


class ViewController:
    """Visualization controller for Scalabel.

    Keymap:
    -  n / p: Show next or previous image
    -  Space: Start / stop animation
    -  t: Toggle 2D / 3D bounding box (if avaliable)
    -  a: Toggle the display of the attribute tags on boxes or polygons.
    -  c: Toggle the display of polygon vertices.
    -  Up: Increase the size of polygon vertices.
    -  Down: Decrease the size of polygon vertices.
    """

    def __init__(
        self,
        config: ControllerConfig,
        display_cfg: DisplayConfig,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        """Initializer."""
        self.config = config
        self.display_cfg = display_cfg
        self.frame_index: int = 0

        # animation
        self._run_animation: bool = False
        self._timer: Timer = Timer(0.4, self.tick)
        self._label_colors: Dict[str, NDArrayF64] = {}

        # load label file
        if not os.path.exists(config.label_path):
            logger.error("Label file not found!")
        self.frames = load(config.label_path, config.nproc).frames[:10]
        logger.info("Load images: %d", len(self.frames))

        self.images: Dict[str, "concurrent.futures.Future[NDArrayU8]"] = {}
        # Cache the images in separate threads.
        for frame in self.frames:
            self.images[frame.name] = executor.submit(
                fetch_image, (frame, self.config.image_dir)
            )
        self.queue: Queue[DisplayData] = Queue()

    def run(self) -> None:
        """Start the visualization."""
        if self.config.out_dir is None:
            plt.connect("key_release_event", self.key_press)
            self.update()
            plt.show()
        else:
            os.makedirs(self.config.out_dir, exist_ok=True)
            while self.frame_index < len(self.frames):
                self.update(output=True)
                self.frame_index += 1

    def update(self, output: bool = False) -> None:
        """Update display_cfg, and put viewer task to the queue."""
        frame = self.frames[self.frame_index % len(self.frames)]
        image = self.images[frame.name].result()
        if not output:
            self.queue.put(DisplayData(image, frame, self.display_cfg))
            return
        assert self.config.out_dir is not None
        out_path = os.path.join(
            self.config.out_dir, frame.name.replace(".jpg", ".png")
        )
        plt.cla()
        self.queue.put(
            DisplayData(image, frame, self.display_cfg, out_path=out_path)
        )

    def key_press(self, event: Event) -> None:
        """Handel control keys."""
        if event.key == "n":
            self.frame_index = (self.frame_index + 1) % len(self.frames)
        elif event.key == "p":
            if self.frame_index == 0:
                self.frame_index = len(self.frames) - 1
            else:
                self.frame_index -= 1
        elif event.key == "t":
            self.display_cfg.with_box2d = not self.display_cfg.with_box2d
            self.display_cfg.with_box3d = not self.display_cfg.with_box3d
        elif event.key == "space":
            if not self._run_animation:
                self.start_animation()
            else:
                self.stop_animation()
            return
        elif event.key == "a":
            self.display_cfg.with_tags = not self.display_cfg.with_tags
        # Control event.keys for polygon mode
        elif event.key == "c":
            self.display_cfg.with_ctrl_points = (
                not self.display_cfg.with_ctrl_points
            )
        elif event.key == "up":
            self.display_cfg.ctrl_point_size = min(
                15.0, self.display_cfg.ctrl_point_size + 0.5
            )
        elif event.key == "down":
            self.display_cfg.ctrl_point_size = max(
                0.0, self.display_cfg.ctrl_point_size - 0.5
            )
        else:
            return

        self.frame_index = max(self.frame_index, 0)
        self.update()

    def tick(self) -> None:
        """Animation tick."""
        self._run_animation = False
        self.start_animation()
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.update()

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
