"""Convert poly2d to rle."""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from typing import Callable, List

from tqdm import tqdm

from ..common.logger import logger
from ..common.parallel import NPROC
from .io import group_and_sort, load, load_label_config, save
from .transforms import frame_to_rles, rle_to_box2d
from .typing import Config, Frame, ImageSize

ToRLEsFunc = Callable[[List[Frame], str, Config, int], None]


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="poly2d/mask to rle format")
    parser.add_argument(
        "-i",
        "--input",
        help=(
            "root directory of scalabel Json files or path to a label "
            "json file"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="path to save rle formatted label file",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="configuration file",
    )
    parser.add_argument(
        "--per-video",
        action="store_true",
        help="store seg_track annotations per video",
    )
    return parser.parse_args()


def frame_to_rle(shape: ImageSize, frame: Frame) -> Frame:
    """Converting a frame of poly2ds to rle."""
    labels = frame.labels
    if labels is None or len(labels) == 0:
        return frame
    # higher score, rendering later
    has_score = all((label.score is not None for label in labels))
    if has_score:
        labels = sorted(
            labels, key=lambda label: float(label.score)  # type: ignore
        )
    filt_labels, poly2ds = [], []
    for label in labels:
        if label.poly2d is None:
            continue
        for p in label.poly2d:
            p.closed = True
        filt_labels.append(label)
        poly2ds.append(label.poly2d)
    poly2ds = [label.poly2d for label in labels if label.poly2d is not None]
    rles = frame_to_rles(shape, poly2ds)
    for label, rle in zip(filt_labels, rles):
        label.poly2d = None
        label.rle = rle
        label.box2d = rle_to_box2d(rle)
    frame.labels = filt_labels
    return frame


def frames_to_rle(
    shapes: List[ImageSize],
    frames: List[Frame],
    nproc: int = NPROC,
) -> List[Frame]:
    """Execute the rle conversion in parallel."""
    if nproc > 1:
        with Pool(nproc) as pool:
            frames = pool.starmap(
                partial(frame_to_rle),
                tqdm(
                    zip(shapes, frames),
                    total=len(frames),
                ),
            )
    else:
        frames = [
            frame_to_rle(shape, frame)
            for shape, frame in tqdm(zip(shapes, frames), total=len(frames))
        ]

    sorted(frames, key=lambda frame: frame.name)
    return frames


def seg_to_rles(
    frames: List[Frame], config: Config, nproc: int = NPROC
) -> List[Frame]:
    """Converting segmentation poly2d to rles."""
    img_shape = config.imageSize
    assert img_shape is not None, "Seg conversion requires imageSize in config"
    img_shapes = [img_shape] * len(frames)
    logger.info("Start conversion for Seg to RLEs")
    return frames_to_rle(img_shapes, frames, nproc)


def main() -> None:
    """Main function."""
    args = parse_args()
    dataset = load(args.input, args.nproc)
    frames, config = dataset.frames, dataset.config

    if args.config is not None:
        config = load_label_config(args.config)
    if config is None:
        raise ValueError(
            "Dataset config is not specified. Please use --config"
            " to specify a config for this dataset."
        )

    frames = seg_to_rles(frames, config, args.nproc)
    if args.per_video:
        frames_list = group_and_sort(frames)
        os.makedirs(args.output, exist_ok=True)
        for video_anns in frames_list:
            video_name = video_anns[0].videoName
            out_path = os.path.join(args.output, f"{video_name}.json")
            save(out_path, video_anns)
    else:
        save(args.output, frames)

    logger.info("Finished!")


if __name__ == "__main__":
    main()
