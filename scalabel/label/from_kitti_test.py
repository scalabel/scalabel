"""Test cases for from_kitti.py."""
import os
import shutil
from argparse import Namespace
from typing import List

import pytest

from ..common.parallel import NPROC
from ..unittest.util import get_test_file
from .from_kitti import run
from .io import load
from .typing import Frame


def compare_results(result: List[Frame], result_compare: List[Frame]) -> None:
    """Compare two list of frames."""
    for frame, frame_ref in zip(result, result_compare):
        assert frame.name == frame_ref.name
        assert frame.video_name == frame_ref.video_name
        assert frame.frame_index == frame_ref.frame_index
        if frame.intrinsics is not None:
            assert frame_ref.intrinsics is not None
            assert frame.intrinsics.focal == pytest.approx(
                frame_ref.intrinsics.focal
            )
            assert frame.intrinsics.center == pytest.approx(
                frame_ref.intrinsics.center
            )
            assert frame.intrinsics.skew == pytest.approx(
                frame_ref.intrinsics.skew
            )
        else:
            assert frame.intrinsics == frame_ref.intrinsics
        if frame.extrinsics is not None:
            assert frame_ref.extrinsics is not None
            assert frame.extrinsics.location == pytest.approx(
                frame_ref.extrinsics.location
            )
            assert frame.extrinsics.rotation == pytest.approx(
                frame_ref.extrinsics.rotation
            )
        else:
            assert frame.extrinsics == frame_ref.extrinsics

        if frame.labels is not None:
            assert frame_ref.labels is not None
            for label, label_ref in zip(frame.labels, frame_ref.labels):
                assert label.id == label_ref.id
                assert label.category == label_ref.category
                if label.box2d is not None:
                    assert label_ref.box2d is not None
                    assert label.box2d.x1 == pytest.approx(label_ref.box2d.x1)
                    assert label.box2d.y1 == pytest.approx(label_ref.box2d.y1)
                    assert label.box2d.x2 == pytest.approx(label_ref.box2d.x2)
                    assert label.box2d.y2 == pytest.approx(label_ref.box2d.y2)
                else:
                    assert label.box2d == label_ref.box2d

                if label.box3d is not None:
                    assert label_ref.box3d is not None
                    assert label.box3d.location == pytest.approx(
                        label_ref.box3d.location
                    )
                    assert label.box3d.dimension == pytest.approx(
                        label_ref.box3d.dimension
                    )
                    assert label.box3d.orientation == pytest.approx(
                        label_ref.box3d.orientation
                    )
                else:
                    assert label.box3d == label_ref.box3d
        else:
            assert frame.labels == frame_ref.labels


def test_run() -> None:
    """Test KITTI conversion."""
    filepath = get_test_file("kitti_sample")
    result_filepath = get_test_file("kitti_tracking.json")
    output_dir = "./processed/"

    args = Namespace(
        input_dir=filepath,
        output_dir=output_dir,
        mode="mini",
        data_type="tracking",
        nproc=NPROC,
    )
    run(args)

    result = load(os.path.join(output_dir, "tracking_mini.json")).frames
    result_compare = load(result_filepath).frames

    compare_results(result, result_compare)

    # clean up
    shutil.rmtree(output_dir)
