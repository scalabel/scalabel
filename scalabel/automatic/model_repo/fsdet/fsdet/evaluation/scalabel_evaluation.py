from functools import partial
from genericpath import exists
import os
import json
from pprint import pp

from detectron2.data import MetadataCatalog
from fsdet.evaluation.evaluator import DatasetEvaluator
from fsdet.utils.file_io import PathManager
from scalabel.common.logger import logger
from scalabel.common.parallel import NPROC, pmap
from scalabel.label.io import load, load_label_config, parse
from scalabel.eval.detect import evaluate_det


class ScalabelEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir): # initial needed variables
        self._dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self._output_dir = output_dir

        if not os.path.isdir(self._output_dir):
            os.makedirs(self._output_dir)

    def reset(self): # reset predictions
        self._predictions = []

    def process(self, inputs, outputs): # prepare predictions for evaluation
        for input, output in zip(inputs, outputs):
            prediction = {
                "name": input["image_id"],
                "url": None,
                "videoName": None,
                "intrinsics": None,
                "extrinsics": None,
                "attributes": None,
                "timestamp": None,
                "frameIndex": None,
                "size": None,
            }
            if "instances" in output:
                instances = output["instances"]
                labels = []
                count = 0
                for box, score, pred_class in zip([box for box in instances.get("pred_boxes")],
                                                  instances.get("scores").tolist(),
                                                  instances.get("pred_classes").tolist()):
                    label = {
                        "id": input["image_id"] + "_" + str(count),
                        "index": None,
                        "manualShape": None,
                        "manualAttributes": None,
                        "score": score,
                        "attributes": None,
                        "category": self._metadata.thing_classes[pred_class],
                        "box2d": {
                            "x1": box[0].cpu().numpy().tolist(),
                            "y1": box[1].cpu().numpy().tolist(),
                            "x2": box[2].cpu().numpy().tolist(),
                            "y2": box[3].cpu().numpy().tolist()
                        },
                        "box3d": None,
                        "poly2d": None,
                        "rle": None,
                        "graph": None
                    }
                    labels.append(label)
                    count += 1
                prediction["labels"] = labels
            self._predictions.append(prediction)

    def save_predictions(self): # save predictions
        with open(PathManager.get_local_path(self._output_dir + "/predictions.json"), "w") as f:
            json.dump(self._predictions, f)
        # pp(self._predictions)

    def load_gts(self):
        gt_path = PathManager.get_local_path(self._metadata.json_file)
        cfg_path = PathManager.get_local_path(self._metadata.cfg)
        cfg = load_label_config(cfg_path)
        dataset = load(gt_path, NPROC)

        return dataset.frames, cfg

    def evaluate(self): # evaluate predictions
        # TODO: call evaluation function from scalabel
        parse_ = partial(parse, validate_frames=True)
        if NPROC > 1:
            frames = pmap(parse_, self._predictions, NPROC)
        else:
            frames = list(map(parse_, self._predictions))

        gts, cfg = self.load_gts()

        if cfg is None:
            raise ValueError(
                "Dataset config is not specified. Please specify a config for this dataset."
            )

        # self.save_predictions()
        # preds_path = PathManager.get_local_path(f"{self._output_dir}/predictions.json")
        # preds = load(preds_path).frames

        eval_result = evaluate_det(gts, frames, cfg, NPROC)
        logger.info(eval_result)

        with open(f"{self._output_dir}/metrics.json", "w") as f:
            json.dump(eval_result.json(), f)

        return {}
