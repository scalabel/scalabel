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

import ray
from ray import serve


@serve.deployment(ray_actor_options={"num_cpus": 32, "num_gpus": 1})
class RayModel(object):
    def __init__(self, cfg, logger):
        self.logger = logger

        self.cfg = cfg.clone()
        self.model = build_model(cfg)

        checkpointer = DetectionCheckpointer(self.model, save_to_disk=True)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.cuda().eval()

        if self.cfg.TASK_TYPE == "OD":
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
        if self.cfg.TASK_TYPE == "OD":
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
        else:
            if request_type == QueryConsts.QUERY_TYPES["inference"]:
                results = await self.handle_batch_polygon(inputs)

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
