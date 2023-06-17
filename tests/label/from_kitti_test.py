"""Test cases for from_kitti.py."""
import os
import shutil
from argparse import Namespace

from scalabel.common.parallel import NPROC
from tests.util import get_test_file
from scalabel.label.from_kitti import run
from scalabel.label.io import load
from scalabel.label.test_utils import compare_groups_results, compare_results


def test_run() -> None:
    """Test KITTI conversion."""
    filepath = get_test_file("kitti_sample")
    result_filepath = get_test_file("kitti_tracking.json")
    output_dir = "./processed/"

    args = Namespace(
        input_dir=filepath,
        output_dir=output_dir,
        split="training",
        data_type="tracking",
        nproc=NPROC,
    )
    run(args)

    result = load(os.path.join(output_dir, "tracking_training.json"))
    result_compare = load(result_filepath)

    compare_results(result.frames, result_compare.frames)
    compare_groups_results(result.groups, result_compare.groups)

    # clean up
    shutil.rmtree(output_dir)
