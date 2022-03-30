import torch
from torch import nn
import numpy as np
import skimage.transform as transform
from collections import OrderedDict
from scalabel.automatic.model_repo.dd3d.structures.boxes3d import Boxes3D
from scalabel.automatic.tools.dd3d_to_synscapes import dd3d_to_cityscapes, kitti_label_to_synscapes
from scalabel.automatic.tools.synscapes_to_cityscapes import synscapes_to_cityscapes
import seaborn as sns
import cv2
import json

import ray
from ray import serve

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog

from scalabel.automatic.model_repo import add_general_config, add_dd3d_config
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti
from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import draw_boxes3d_cam
from scalabel.automatic.model_repo.dd3d.utils.visualization import float_to_uint8_color
from scalabel.automatic.models.ray_model_new import RayModel

import os
from PIL import Image
import torchvision.transforms.functional as TF

VALID_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=8)]
COLORMAP = OrderedDict(
    {
        "Car": COLORS[2],  # green
        "Pedestrian": COLORS[1],  # orange
        "Cyclist": COLORS[0],  # blue
        "Van": COLORS[6],  # pink
        "Truck": COLORS[5],  # brown
        "Person_sitting": COLORS[4],  #  purple
        "Tram": COLORS[3],  # red
        "Misc": COLORS[7],  # gray
    }
)


def predict_synscapes(model, num):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    image_path = os.path.join(dir_path, f"/scratch-second/egroenewald/synscapes/Synscapes/img/rgb/{num}.png")
    meta_path = os.path.join(dir_path, f"/scratch-second/egroenewald/synscapes/Synscapes/meta/{num}.json")
    gt_path = os.path.join(dir_path, f"../../../local-data/synscapes-eval/gt/{num}.json")
    pred_path = os.path.join(dir_path, f"../../../local-data/synscapes-eval/pred/{num}.json")

    img = np.array(Image.open(image_path))
    image_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    intrinsics = metadata["camera"]["intrinsic"]
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    u0 = intrinsics["u0"]
    v0 = intrinsics["v0"]

    intrinsics = torch.tensor([[fx, 0.0000, u0], [0.0000, fy, v0], [0.0000, 0.0000, 1.0000]]).cpu().numpy()
    batched_inputs = [
        {
            "image": image_tensor,
            "intrinsics": torch.tensor(intrinsics.tolist()),
        }
    ]
    print(f"Predicting {num}")
    with torch.no_grad():
        output = model(batched_inputs)
    instances = output[0]["instances"]

    print(f"Writing pred file for {num}")
    synscapes_labels = dd3d_to_cityscapes(instances)
    with open(pred_path, "w") as file:
        json.dump(synscapes_labels, file)

    print(f"Writing gt file for {num}")
    get_labels = synscapes_to_cityscapes(metadata)
    with open(gt_path, "w") as file:
        json.dump(get_labels, file)


def predict():
    cfg = get_cfg()
    add_general_config(cfg)
    add_dd3d_config(cfg)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "../model_repo/configs/dd3d/dd3d.yaml")
    print("file_path:", file_path)
    cfg.merge_from_file(file_path)

    cfg.TASK_TYPE = "box3d"
    cfg.MODEL.WEIGHTS = os.path.join(dir_path, "../../..", cfg.MODEL.WEIGHTS)
    cfg.MODEL.CKPT = os.path.join(dir_path, "../../..", cfg.MODEL.CKPT)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_to_disk=True)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.cuda().eval()

    for i in range(1, 1000):
        predict_synscapes(model, i)


if __name__ == "__main__":
    predict()
