import argparse
import os
import json
from urllib import request
from redis import Redis
from pprint import pformat
from tqdm import tqdm

from detectron2.data.detection_utils import read_image
from scalabel_bot.models.few_shot_detection.fsdet.config import (
    get_cfg,
)
from scalabel_bot.models.few_shot_detection.fsdet.engine import (
    DefaultPredictor,
)

from scalabel_bot.common.consts import (
    REDIS_HOST,
    REDIS_PORT,
)

from scalabel.common.logger import logger


MODEL_NAME = "FSDET"


class FSDET:
    def __init__(self):
        self.args = self.get_parser().parse_args(
            [
                "--config-file",
                "scalabel_bot/models/few_shot_detection/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml",
                "--opts",
                "MODEL.WEIGHTS",
                "fsdet://coco/tfa_cos_1shot/model_final.pth",
            ]
        )
        # logger.info(f"Arguments: {str(self.args)}")
        self._data_loader = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            encoding="utf-8",
            decode_responses=True,
        )

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.freeze()
        return cfg

    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="FsDet demo for builtin models"
        )
        parser.add_argument(
            "--config-file",
            default=(
                "configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
            ),
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--webcam", action="store_true", help="Take inputs from webcam."
        )
        parser.add_argument("--video-input", help="Path to video file.")
        parser.add_argument(
            "--input",
            nargs="+",
            help=(
                "A list of space separated input images; "
                "or a single glob pattern such as 'directory/*.jpg'"
            ),
        )
        parser.add_argument(
            "--output",
            help=(
                "A file or directory to save output visualizations. "
                "If not given, will show output in an OpenCV window."
            ),
        )

        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help=(
                "Modify config options using the command-line 'KEY VALUE' pairs"
            ),
            default=[],
            nargs=argparse.REMAINDER,
        )
        return parser

    def import_data(self, task):
        data_str = self._data_loader.get(task["taskKey"])
        data = json.loads(data_str)
        img_urls = [item["urls"]["-1"] for item in data["task"]["items"]]
        imgs = []
        for img_url in tqdm(
            img_urls,
            desc="Retrieving images",
            leave=True,
            position=0,
            unit="items",
        ):
            img_name = img_url.split("/")[-1]
            img_dir = os.path.join("data", task["projectName"])
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                os.makedirs(img_dir, exist_ok=True)
                request.urlretrieve(img_url, img_path)
            img = read_image(img_path, format="BGR")
            imgs.append(img)
        return imgs

    def import_model(self, device=None):
        cfg = self.setup_cfg(self.args)
        if device is not None:
            cfg.defrost()
            cfg.MODEL.DEVICE = device
            cfg.freeze()
        predictor = DefaultPredictor(cfg)
        return predictor
