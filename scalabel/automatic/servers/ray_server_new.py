import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import redis
import json
import time
import torch
import torch.multiprocessing as mp

import ray
from ray import serve

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from scalabel.automatic.models.ray_model_new import RayModel
import scalabel.automatic.consts.redis_consts as RedisConsts
import scalabel.automatic.consts.query_consts as QueryConsts

ray.init()
serve.start(http_options={"port": 8001})


class RayModelServerScheduler(object):
    def __init__(self, server_config, model_registry_config, logger):
        self.logger = logger

        self.server_config = server_config
        self.model_registry_config = model_registry_config

        self.redis = redis.Redis(host=server_config["redis_host"], port=server_config["redis_port"])

        self.model_register_channel = RedisConsts.REDIS_CHANNELS["modelRegister"]
        self.model_request_channel = RedisConsts.REDIS_CHANNELS["modelRequest"]
        self.model_response_channel = RedisConsts.REDIS_CHANNELS["modelResponse"]
        self.model_kill_channel = RedisConsts.REDIS_CHANNELS["modelKill"]

        self.task_configs = {}
        self.task_models = {}
        self.task_images = {}

        self.threads = {}

        self.verbose = False

        self.restore()
        self.logger.info("Model server launched.")

    # restore when server restarts, connects to redis channels.
    def restore(self):
        task_names = self.redis.smembers("ModelServerTasks")
        for task_name in task_names:
            task_name = task_name.decode()
            task_config = json.loads(self.redis.get(task_name))
            if task_config["active"]:
                self.restore_model(task_name, task_config)
                self.restore_image(task_name, task_config)
            else:
                continue
            self.task_configs[task_name] = task_config

            model_request_channel = self.model_request_channel % task_name
            model_request_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
            model_request_subscriber.subscribe(**{model_request_channel: self.request_handler})
            thread = model_request_subscriber.run_in_thread(sleep_time=0.001)
            self.threads[model_request_channel] = thread

            self.logger.info(f"Restore model for {task_name}")

    def restore_model(self, task_name, task_config):
        model_dir = task_config["model_dir"]
        model_cfg = task_config["model_cfg"]
        cfg = CfgNode(model_cfg)
        cfg.MODEL.WEIGHTS = model_dir

        num_replicas = task_config["deploy_config"]["num_replicas"]
        RayModel.options(name=task_name, num_replicas=num_replicas).deploy(
            cfg,
            self.logger,
        )

        deploy_model = serve.get_deployment(task_name).get_handle()
        self.task_models[task_name] = deploy_model

    def restore_image(self, task_name, task_config):
        image_dir = task_config["image_dir"]
        self.task_images[task_name] = torch.load(image_dir)

    # save the loaded tasks
    def save(self, task_name):
        self.save_config(task_name)
        self.save_model(task_name)
        self.save_image(task_name)

    def save_config(self, task_name):
        task_config = json.dumps(self.task_configs[task_name])
        self.redis.set(task_name, task_config)

    def save_model(self, task_name):
        task_config = self.task_configs[task_name]
        task_model = self.task_models[task_name]

        task_model.save.remote(task_config["model_dir"])

    def save_image(self, task_name):
        task_config = self.task_configs[task_name]
        task_images = self.task_images[task_name]

        torch.save(task_images, task_config["image_dir"])

    def listen(self):
        model_register_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_register_subscriber.subscribe(**{self.model_register_channel: self.register_handler})
        thread_register = model_register_subscriber.run_in_thread(sleep_time=0.001)

        self.threads[self.model_register_channel] = thread_register

        model_kill_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_kill_subscriber.subscribe(**{self.model_kill_channel: self.kill_handler})
        thread_kill = model_kill_subscriber.run_in_thread(sleep_time=0.001)

        self.threads[self.model_kill_channel] = thread_kill

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

            task_images = self.task_images[f'{project_name}_{task_id}']
            inputs = [task_images[item["url"]] for item in request_data["items"]]

            model = self.task_models[f'{project_name}_{task_id}']
            model.remote(inputs, request_data, request_type)

    def kill_handler(self, kill_message):
        kill_message = json.loads(kill_message["data"])
        project_name = kill_message["projectName"]
        task_id = kill_message["taskId"]
        task_name = f'{project_name}_{task_id}'

        self.task_configs[task_name]["active"] = False
        self.save_config(task_name)

        model = self.task_models[task_name]
        model.idle.remote()

        self.logger.info(f"{task_name} recevied no action for a period. Set to idle.")

    def register_task(self, task_type, project_name, task_id, item_list):
        task_name = f"{project_name}_{task_id}"
        if task_name in self.task_configs:
            if not self.task_configs[task_name]["active"]:
                self.task_configs[task_name]["active"] = True
                self.task_models[task_name].activate.remote()
                self.save(task_name)
            return
        elif self.redis.smembers("ModelServerTasks") != None and task_name in self.redis.smembers("ModelServerTasks"):
            self.task_configs[task_name] = json.loads(self.redis.get(task_name))
            self.task_configs[task_name]["active"] = True

            self.restore_model(task_name, self.task_configs[task_name])
            self.restore_image(task_name, self.task_configs[task_name])
        else:
            model_registry_config = self.model_registry_config[task_type]

            model_name = model_registry_config["models"][model_registry_config["defaults"]["model"]]
            deploy_config = model_registry_config["defaults"]["config"]

            self.initialize(task_name, model_name, deploy_config, item_list)

            self.redis.sadd("ModelServerTasks", task_name)
            self.save(task_name)

        model_request_channel = self.model_request_channel % task_name
        model_request_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_request_subscriber.subscribe(**{model_request_channel: self.request_handler})
        thread = model_request_subscriber.run_in_thread(sleep_time=0.001)
        self.threads[model_request_channel] = thread

    def initialize(self, task_name, model_name, deploy_config, item_list):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

        num_replicas = deploy_config["num_replicas"]
        RayModel.options(name=task_name, num_replicas=num_replicas).deploy(
            cfg,
            self.logger,
        )

        deploy_model = serve.get_deployment(task_name).get_handle()
        self.task_configs[task_name] = {
            "task_name": task_name,
            "deploy_config": deploy_config,
            "item_list": item_list,
            "model_cfg": cfg,
            "model_dir": task_name + ".pth",
            "image_dir": task_name + "_image.pkl",
            "active": True
        }
        self.task_models[task_name] = deploy_model

        self.task_images[task_name] = ray.get(deploy_model.load_inputs.remote(item_list))

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
    model_registry_config = {
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

    scheduler = RayModelServerScheduler(server_config, model_registry_config, logger)
    scheduler.listen()


if __name__ == "__main__":
    launch()
