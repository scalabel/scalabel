import os
from urllib import request
from io import BytesIO
from PIL import Image
import numpy as np
import time
import copy
import json

import torch
from torch.multiprocessing import Pool

from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.config import get_cfg
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer

from scalabel.automatic.scalabel_bot.common.logger import logger
from scalabel.automatic.model_repo.dd3d.utils.convert import (
    convert_3d_box_to_kitti,
)
from scalabel.automatic.model_repo.dd3d.config import add_dd3d_config


MODEL_NAME = "DD3D"


class DD3D:
    def __init__(self):
        self.model_config = (
            "scalabel/automatic/model_repo/configs/dd3d/dd3d.yaml"
        )
        self.cfg = None
        self.model = None

    def import_model(self, device):
        cfg = get_cfg()
        cfg.TASK_TYPE = "box3d"
        cfg.MODEL.DEVICE = f"cuda:{device}"
        add_dd3d_config(cfg)
        cfg.merge_from_file(self.model_config)
        self.cfg = cfg.clone()
        self.model = build_model(cfg)
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=True)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.cuda().eval()

    @staticmethod
    def url_to_img(url, img_dir):
        img_name = url.split("/")[-1]
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            os.makedirs(img_dir, exist_ok=True)
            request.urlretrieve(url, img_path)
        img = read_image(img_path, format="BGR")
        height, width = img.shape[:2]
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return {"image": img, "height": height, "width": width}

    def import_data(self, task):
        image_list = []
        img_dir = os.path.join(
            "data", f"{task['projectName']}_{task['taskId']}"
        )
        for item in task["items"]:
            img = self.url_to_img(item["url"], img_dir)
            if task["items"][0]["intrinsics"] is not None:
                intrinsics = item["intrinsics"]
                intrinsics_tensor = torch.Tensor(
                    [
                        [intrinsics["focal"][0], 0.0, intrinsics["center"][0]],
                        [0.0, intrinsics["focal"][1], intrinsics["center"][1]],
                        [0.0, 0.0, 1.0],
                    ]
                )
                img.update({"intrinsics": intrinsics_tensor})
            image_list.append(img)
        return image_list

    def handle_batch_detection(self, inputs):
        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions

    def __call__(self, inputs):
        results = self.handle_batch_detection(inputs)
        instances = results[0]["instances"]
        boxes3d = []
        for pred_box3d in instances.pred_boxes3d:
            w, l, h, x, y, z, rot_y, _ = convert_3d_box_to_kitti(pred_box3d)
            y = y - h / 2  # Add half height to get center
            converted_box = [
                w,
                l,
                h,
                x,
                y,
                z,
                np.pi / 2,
                0,
                np.pi / 2 - rot_y,
            ]
            converted_box = [float(v) for v in converted_box]
            boxes3d.append(converted_box)

        # Filter only car labels
        result = []

        def interval_overlaps(a, b):
            return min(a[1], b[1]) - max(a[0], b[0]) > 0

        def overlaps(box1, box2):
            b1w, b1l, b1h, b1x, b1y, b1z = box1[:6]
            b2w, b2l, b2h, b2x, b2y, b2z = box2[:6]
            return (
                interval_overlaps(
                    (b1x - b1w / 2, b1x + b1w / 2),
                    (b2x - b2w / 2, b2x + b2w / 2),
                )
                and interval_overlaps(
                    (b1y - b1h / 2, b1y + b1h / 2),
                    (b2y - b2h / 2, b2y + b2h / 2),
                )
                and interval_overlaps(
                    (b1z - b1l / 2, b1z + b1l / 2),
                    (b2z - b2l / 2, b2z + b2l / 2),
                )
            )

        # Filter boxes too close to origin
        boxes3d = [box for box in boxes3d if np.linalg.norm(box[3:6]) > 4]

        # Filter overlapping boxes
        boxes3d.sort(key=lambda box: (np.linalg.norm(box[3:6]), box[3]))
        if len(boxes3d) > 1:
            result.append(boxes3d[1])
            for idx in range(1, len(boxes3d)):
                if not overlaps(boxes3d[idx], boxes3d[idx - 1]):
                    result.append(boxes3d[idx])

        return {"boxes": result}
