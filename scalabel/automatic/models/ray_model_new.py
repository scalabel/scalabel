import requests
from io import BytesIO
from PIL import Image
import numpy as np
import time
import copy
import json
import redis

import torch
from torch.multiprocessing import Pool

from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.config import get_cfg
from detectron2.utils.events import EventStorage, get_event_storage
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer

import scalabel.automatic.consts.redis_consts as RedisConsts
import scalabel.automatic.consts.query_consts as QueryConsts
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti

import ray
from ray import serve


@serve.deployment(ray_actor_options={"num_cpus": 4, "num_gpus": 1})
class RayModel(object):
    def __init__(self, cfg, logger):
        self.logger = logger

        self.cfg = cfg.clone()
        self.model = build_model(cfg)

        checkpointer = DetectionCheckpointer(self.model, save_to_disk=True)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.cuda().eval()

        if self.cfg.TASK_TYPE == "box2d":
            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
        else:
            self.aug = None

        self.model_response_channel = RedisConsts.REDIS_CHANNELS["modelResponse"]
        self.redis = redis.Redis(host=RedisConsts.REDIS_HOST, port=RedisConsts.REDIS_PORT)

    @staticmethod
    def url_to_img(url, aug):
        img_response = requests.get(url)
        # TODO: convert to bgr
        img = np.array(Image.open(BytesIO(img_response.content)))
        height, width = img.shape[:2]
        if aug is not None:
            img = aug.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return {"image": img, "height": height, "width": width}

    def load_inputs(self, item_list):
        urls = [item["urls"]["-1"] for item in item_list]
        image_list = [self.url_to_img(url, self.aug) for url in urls]

        image_dict = {}
        for url, image in zip(urls, image_list):
            image_dict[url] = image
        return image_dict

    @serve.batch(max_batch_size=1)
    async def handle_batch_detection(self, inputs):
        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions

    @serve.batch(max_batch_size=1)
    async def handle_batch_polygon(self, inputs):
        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions

    # 0 for inference, 1 for training
    async def __call__(self, inputs, request_data, request_type):
        # inference
        if self.cfg.TASK_TYPE == "box2d":
            if request_type == QueryConsts.QUERY_TYPES["inference"]:
                results = await self.handle_batch_detection(inputs)

                model_response_channel = self.model_response_channel % request_data["name"]

                pred_boxes = []
                for box in results["instances"].pred_boxes:
                    box = box.cpu().numpy()
                    pred_boxes.append(box.tolist())
                item_indices = request_data["item_indices"]
                action_packet_id = request_data["action_packet_id"]
                self.redis.publish(model_response_channel, json.dumps([pred_boxes, item_indices, action_packet_id]))
        elif self.cfg.TASK_TYPE == "box3d":
            if request_type == QueryConsts.QUERY_TYPES["inference"]:
                results = await self.handle_batch_detection(inputs)

                # self.logger.info(f"results {results}")
                print("results", results)

                model_response_channel = self.model_response_channel % request_data["name"]

                # boxes3d = results["instances"].pred_boxes3d.vectorize().cpu().numpy().tolist()
                # boxes3d = [[float(v) for v in list(convert_3d_box_to_kitti(box))] for box in instances.pred_boxes3d]
                instances = results["instances"]
                boxes3d = []
                for pred_box3d in instances.pred_boxes3d:
                    w, l, h, x, y, z, rot_y, _ = convert_3d_box_to_kitti(pred_box3d)
                    y = y - h / 2  # Add half height to get center
                    converted_box = [w, l, h, x, y, z, np.pi / 2, 0, np.pi / 2 - rot_y]
                    converted_box = [float(v) for v in converted_box]
                    boxes3d.append(converted_box)

                # Filter only car labels
                result = []
                # for idx, pred_class in enumerate(instances.pred_classes):
                #     if pred_class == 0:
                #         result.append(boxes3d[idx])

                def interval_overlaps(a, b):
                    return min(a[1], b[1]) - max(a[0], b[0]) > 0

                def overlaps(box1, box2):
                    b1w, b1l, b1h, b1x, b1y, b1z = box1[:6]
                    b2w, b2l, b2h, b2x, b2y, b2z = box2[:6]
                    return (
                        interval_overlaps((b1x - b1w / 2, b1x + b1w / 2), (b2x - b2w / 2, b2x + b2w / 2))
                        and interval_overlaps((b1y - b1h / 2, b1y + b1h / 2), (b2y - b2h / 2, b2y + b2h / 2))
                        and interval_overlaps((b1z - b1l / 2, b1z + b1l / 2), (b2z - b2l / 2, b2z + b2l / 2))
                    )

                # Filter boxes too close to origin
                boxes3d = [box for box in boxes3d if np.linalg.norm(box[3:6]) > 4]

                # Filter overlapping boxes
                boxes3d.sort(key=lambda box: (np.linalg.norm(box[3:6]), box[3]))
                result.append(boxes3d[1])
                for idx in range(1, len(boxes3d)):
                    if not overlaps(boxes3d[idx], boxes3d[idx - 1]):
                        result.append(boxes3d[idx])

                boxes = np.array([box.cpu().numpy() for box in results["instances"].pred_boxes]).tolist()
                item_indices = request_data["item_indices"]
                action_packet_id = request_data["action_packet_id"]
                self.redis.publish(model_response_channel, json.dumps([result, item_indices, action_packet_id]))
        else:
            if request_type == QueryConsts.QUERY_TYPES["inference"]:
                results = await self.handle_batch_polygon(inputs)

                self.logger.info(results)

                model_response_channel = self.model_response_channel % request_data["name"]

                item_indices = request_data["item_indices"]
                action_packet_id = request_data["action_packet_id"]
                self.redis.publish(model_response_channel, json.dumps([results, item_indices, action_packet_id]))

    def idle(self):
        self.model.cpu()
        torch.cuda.empty_cache()

    def activate(self):
        self.model.cuda()

    def save(self, dir):
        torch.save({"model": self.model.state_dict()}, dir)
