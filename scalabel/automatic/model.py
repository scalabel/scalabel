from typing import Dict, List

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import time

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


class Predictor:
    def __init__(self, cfg_path: str, item_list: List, num_workers: int, logger) -> None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        # NOTE: you may customize cfg settings
        # cfg.MODEL.DEVICE="cuda" # use gpu by default
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # you can also give a path to you checkpoint
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        self.cfg = cfg.clone()
        self.model = build_model(cfg)

        # These are for training
        self.optimizer = build_optimizer(cfg, self.model)
        self.train_steps = 1

        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.image_dict = {}
        self.load_inputs(item_list, num_workers)

        self.verbose = True

    @staticmethod
    def url_to_img(url: str, aug: object, device: str) -> Dict:
        img_response = requests.get(url)
        img = np.array(Image.open(BytesIO(img_response.content)))
        height, width = img.shape[:2]
        img = aug.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        if device == "cuda":
            img = img.pin_memory()
            img = img.cuda(non_blocking=True)
        return {"image": img, "height": height, "width": width}

    def load_inputs(self, item_list: List, num_workers: int) -> None:
        urls = [item["urls"]["-1"] for item in item_list]
        if num_workers > 1:
            pool = Pool(num_workers)
            image_list = list(pool.starmap(self.url_to_img,
                                           zip(urls,
                                               [self.aug] * len(urls),
                                               [self.cfg.MODEL.DEVICE] * len(urls)
                                               )))
        else:
            image_list = [self.url_to_img(url, self.aug, self.cfg.MODEL.DEVICE) for url in urls]

        for url, image in zip(urls, image_list):
            self.image_dict[url] = image

    # 0 for inference, 1 for training
    def __call__(self, items: Dict, request_type: int = 0) -> List:
        self.calc_time(init=True)
        inputs = [self.image_dict[item["url"]] for item in items]

        # inference
        if request_type == 0:
            with torch.no_grad():
                predictions = self.model(inputs)
                self.calc_time("model inference time")
                return predictions
        # training
        else:
            # define a pseudo field for training simulation
            instance_field = {"gt_boxes": Boxes(torch.tensor([1.0, 1.0, 100.0, 100.0]).unsqueeze(0)),
                              "gt_classes": torch.tensor([0])}

            for i in range(len(inputs)):
                inputs[i]["instances"] = Instances((inputs[i]["height"], inputs[i]["width"]),
                                                   **instance_field)
            self.calc_time("pack training data time")

            self.model.train()
            self.calc_time("changing to training time")
            with EventStorage(0) as self.storage:
                for _ in range(self.train_steps):
                    loss_dict = self.model(inputs)
                    losses = sum(loss_dict.values())

                    self.calc_time("loss calculation time")

                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    self.calc_time("optimization time")
            self.model.eval()
            self.calc_time("changing back to eval time")
            return []

    def calc_time(self, message="", init=False):
        if not self.verbose:
            return
        if init:
            self.start_time = time.time()
        else:
            print(message + ": {}s".format(time.time() - self.start_time))
            self.start_time = time.time()