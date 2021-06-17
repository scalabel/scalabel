from typing import Dict, List

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import torch

from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer


class Predictor:
    def __init__(self, cfg_path: str, num_workers: int = 4) -> None:
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
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        # self.pool = Pool(num_workers)

    def url_to_img(self, url: str) -> Dict:
        img_response = requests.get(url)
        img = np.array(Image.open(BytesIO(img_response.content)))
        height, width = img.shape[:2]
        img = self.aug.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        if self.cfg.MODEL.DEVICE == "cuda":
            img = img.pin_memory()
            img = img.cuda(non_blocking=True)
        return {"image": img, "height": height, "width": width}

    def load_inputs(self, urls: List[str]) -> List[Dict]:
        return [self.url_to_img(url) for url in urls]
        # return list(self.pool.map(self.url_to_img, urls))

    def __call__(self, items: Dict) -> List:
        urls = [item["url"] for item in items]
        inputs = self.load_inputs(urls)
        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
