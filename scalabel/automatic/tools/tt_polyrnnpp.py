import torch
from torch import nn
import numpy as np
import skimage.transform as transform

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model

from scalabel.automatic.model_repo import add_general_config, add_polyrnnpp_config

if __name__ == '__main__':
    cfg = get_cfg()
    add_general_config(cfg)
    add_polyrnnpp_config(cfg)

    model = build_model(cfg)

    batched_inputs = {
        "image": torch.rand(3, 60, 60),
        "bbox": [10, 10, 20, 20],
        "poly": np.random.randint(low=10, high=20, size=[50, 2])
    }

    output = model(batched_inputs)

    for k in output.keys():
        print(k, output[k].size())