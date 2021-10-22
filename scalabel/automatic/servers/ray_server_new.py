import os
import logging
import redis
import json
import time
import torch.multiprocessing as mp

import ray
from ray import serve

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from scalabel.automatic.models.ray_model_new import RayModel
import scalabel.automatic.consts.redis_consts as RedisConsts
import scalabel.automatic.consts.query_consts as QueryConsts

ray.init()
serve.start(http_options={"port": 8001})


class RayModelServerScheduler(object):
    def __init__(self, server_config, model_config, logger):
        self.logger = logger

        self.server_config = server_config
        self.model_config = model_config

        self.redis = redis.Redis(host=server_config["redis_host"], port=server_config["redis_port"])

        self.model_register_channel = RedisConsts.REDIS_CHANNELS["modelRegister"]
        self.model_request_channel = RedisConsts.REDIS_CHANNELS["modelRequest"]
        self.model_response_channel = RedisConsts.REDIS_CHANNELS["modelResponse"]

        self.tasks = {}
        self.threads = {}

        self.models = {}
        self.load_models()

        self.verbose = False

        self.logger.info("Model server launched.")

    def load_models(self):
        valid_tasks = self.model_config["tasks"]
        for task in valid_tasks:
            if task in self.model_config:
                valid_models = self.model_config[task]["models"]
                for model_name in valid_models:
                    cfg = get_cfg()
                    cfg.merge_from_file(model_zoo.get_config_file(valid_models[model_name]))
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(valid_models[model_name])
                    cfg.MODEL.DEVICE = "cpu"

                    model = build_model(cfg)
                    checkpointer = DetectionCheckpointer(model)
                    checkpointer.load(cfg.MODEL.WEIGHTS)
                    self.models[f"{task}_{model_name}"] = {
                        "config": cfg,
                        "model": model
                    }

    # restore when server restarts, connects to redis channels.
    def restore(self):
        pass

    # save the loaded tasks
    def save(self):
        pass

    def listen(self):
        model_register_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_register_subscriber.subscribe(**{self.model_register_channel: self.register_handler})
        thread = model_register_subscriber.run_in_thread(sleep_time=0.001)

        self.threads[self.model_register_channel] = thread

    def register_handler(self, register_message):
        register_message = json.loads(register_message["data"])

        task_type = "OD"
        if "taskType" in register_message:
            task_type = register_message["taskType"]
        project_name = register_message["projectName"]
        task_id = register_message["taskId"]
        item_list = register_message["items"]

        self.register_task(task_type, project_name, task_id, item_list)

        self.logger.info(f"Set up model inference for {project_name}: {task_id}.")

    def request_handler(self, request_message):
        self.calc_time(init=True)

        # decode request message
        request_message = json.loads(request_message["data"])
        project_name = request_message["projectName"]
        task_id = request_message["taskId"]
        items = request_message["items"]
        item_indices = request_message["itemIndices"]
        action_packet_id = request_message["actionPacketId"]
        request_type = request_message["type"]

        if request_type == QueryConsts.QUERY_TYPES["inference"]:
            request_data = {
                "name": f'{project_name}_{task_id}',
                "items": items,
                "item_indices": item_indices,
                "action_packet_id": action_packet_id,
                "request_type": request_type
            }

            self.calc_time("save data to query list")

            model = self.tasks[f'{project_name}_{task_id}']["model"]
            model.remote(request_data, request_type)

    def register_task(self, task_type, project_name, task_id, item_list):
        deploy_name = f"{project_name}_{task_id}"
        if self.tasks[deploy_name]["active"]:
            return

        model_config = self.model_config[task_type]

        model_name = model_config["defaults"]["model"]
        deploy_config = model_config["defaults"]["config"]

        model = self.models[f"{task_type}_{model_name}"]["model"]
        config = self.models[f"{task_type}_{model_name}"]["config"]

        self.setup_model(deploy_name, model, config, deploy_config, item_list)

        model_request_channel = self.model_request_channel % deploy_name
        model_request_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_request_subscriber.subscribe(**{model_request_channel: self.request_handler})
        thread = model_request_subscriber.run_in_thread(sleep_time=0.001)
        self.threads[model_request_channel] = thread

    def setup_model(self, name, model, config, deploy_config, item_list):
        num_replicas = deploy_config["num_replicas"]
        RayModel.options(name=name, num_replicas=num_replicas).deploy(
            model,
            config,
            item_list,
            self.logger,
        )

        deploy_model = serve.get_deployment(name).get_handle()
        self.tasks[name] = {
            "name": name,
            "model": deploy_model,
            "config": deploy_config,
            "active": True
        }

    def close(self):
        for thread_name, thread in self.threads.items():
            thread.stop()

    def calc_time(self, message="", init=False):
        if not self.verbose:
            return
        if init:
            self.start_time = time.time()
        else:
            self.logger.info(message + ": {}s".format(time.time() - self.start_time))
            self.start_time = time.time()


def launch() -> None:
    """Launch processes."""
    log_f = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_f)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # fh = logging.FileHandler('ray_latency_bs_1.txt')
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    # scheduler config
    # create scheduler
    server_config = {
        "redis_host": RedisConsts.REDIS_HOST,
        "redis_port": RedisConsts.REDIS_PORT
    }
    model_config = {
        "tasks": ["OD", "Polygon", "Mask"],
        "OD": {
            "models": {
                "R50-FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "R101-FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            },
            "defaults": {
                "model": "R50-FPN",
                "config": {
                    "bs_infer": 1,
                    "bs_train": 1,
                    "batch_wait_time": 1,  # second
                    "num_replicas": 1
                }
            }
        }
    }

    scheduler = RayModelServerScheduler(server_config, model_config, logger)
    scheduler.listen()


if __name__ == "__main__":
    launch()
