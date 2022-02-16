import torch
from torch import nn
import numpy as np
import skimage.transform as transform

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model

from scalabel.automatic.model_repo import add_general_config, add_dd3d_config
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti
import os


if __name__ == "__main__":
    cfg = get_cfg()
    add_general_config(cfg)
    add_dd3d_config(cfg)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # file_path = os.path.join(dir_path, "../model_repo/dd3d/configs/dd3d.yaml")
    # print("file_path:", file_path)
    # cfg.merge_from_file(file_path, True)

    model = build_model(cfg)

    batched_inputs = [
        {
            "image": torch.rand(3, 60, 60),
            "intrinsics": torch.tensor(
                [[738.9661, 0.0000, 624.2830], [0.0000, 738.8547, 177.0025], [0.0000, 0.0000, 1.0000]]
            ),
            "bbox": [10, 10, 20, 20],
            "poly": np.random.randint(low=10, high=20, size=[50, 2]),
        }
    ]

    model.eval()
    with torch.no_grad():
        output = model(batched_inputs)
    boxes3d = output[0]["instances"].pred_boxes3d
    result = [convert_3d_box_to_kitti(box) for box in boxes3d]

    print("num boxes:", len(result))
    filtered = [box for box in result if box[0] != 0.0 and box[1] != 0.0 and box[2] != 0]
    print("num filtered:", len(filtered))
    print("filtered", filtered)
    # print("Output", result)
