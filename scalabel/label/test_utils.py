"""Utils for from_waymo_test.py and from_kitti_test.py."""
from typing import List, Optional

import pytest

from .typing import Frame, FrameGroup


def compare_results(result: List[Frame], result_compare: List[Frame]) -> None:
    """Compare two list of frames."""
    for frame, frame_ref in zip(result, result_compare):
        assert frame.name == frame_ref.name
        assert frame.videoName == frame_ref.videoName
        assert frame.frameIndex == frame_ref.frameIndex
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
            if frame_ref.labels is None:
                frame_ref.labels = []
                assert len(frame.labels) == 0

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


def compare_groups_results(
    result: Optional[List[FrameGroup]],
    result_compare: Optional[List[FrameGroup]],
) -> None:
    """Compare two list of group of frames."""
    assert result is not None and result_compare is not None
    for group, group_ref in zip(result, result_compare):
        assert group.name == group_ref.name
        assert group.videoName == group_ref.videoName
        assert group.url == group_ref.url
        if group.extrinsics is not None:
            assert group_ref.extrinsics is not None
            assert group.extrinsics.location == pytest.approx(
                group_ref.extrinsics.location
            )
            assert group.extrinsics.rotation == pytest.approx(
                group_ref.extrinsics.rotation
            )
        else:
            assert group.extrinsics == group_ref.extrinsics

        if group.frames is not None:
            assert group_ref.frames is not None
            for frame_names, frame_names_ref in zip(
                group.name, group_ref.name
            ):
                assert frame_names == frame_names_ref
        else:
            assert group.frames == group_ref.frames

        if group.labels is not None:
            if group_ref.labels is None:
                group_ref.labels = []
                assert len(group.labels) == 0

            for label, label_ref in zip(group.labels, group_ref.labels):
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
