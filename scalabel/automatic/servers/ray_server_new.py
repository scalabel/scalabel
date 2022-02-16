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
from scalabel.automatic.consts import ModelStatus

from scalabel.automatic.model_repo import add_general_config, add_polyrnnpp_config, add_dd3d_config

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
        self.model_status_channel = RedisConsts.REDIS_CHANNELS["modelStatus"]
        self.model_notify_channel = RedisConsts.REDIS_CHANNELS["modelNotify"]

        self.task_configs = {}
        self.task_models = {}
        self.task_images = {}

        self.threads = {}

        self.logger.info("Model server launched.")
        self.restore()

    def setup_handler_for_channel(self, channel, handler):
        subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        subscriber.subscribe(**{channel: handler})
        thread = subscriber.run_in_thread(sleep_time=0.001)
        self.threads[channel] = thread

    # get task names in redis
    @property
    def redis_task_names(self):
        return list(map(lambda x: x.decode(), self.redis.smembers("ModelServerTasks")))

    def get_task_config(self, task_name):
        if task_name in self.task_configs:
            return self.task_configs[task_name]
        else:
            return json.loads(self.redis.get(task_name))

    # restore when server restarts, connects to redis channels.
    def restore(self):
        for task_name in self.redis_task_names:
            if task_name in ["a-poly-31_000000", "test-bot_0"]:
                continue
            task_config = self.get_task_config(task_name)

            # if it is not a active task, do not restore it
            if not task_config["active"] or task_name in self.task_configs:
                continue

            # send notification message to let them know the model is loading
            model_notify_channel = self.model_notify_channel % task_name
            self.redis.publish(model_notify_channel, ModelStatus.LOADING.value)

            self.logger.info(f"Restoring model for task: {task_name}")
            self.restore_model(task_name, task_config)
            self.restore_image(task_name, task_config)
            self.task_configs[task_name] = task_config

            model_request_channel = self.model_request_channel % task_name
            self.logger.info(f"Setting up handler for channel {model_request_channel}")
            self.setup_handler_for_channel(model_request_channel, self.request_handler)

            # send notification message to let them know the model is ready
            self.redis.publish(model_notify_channel, ModelStatus.READY.value)
            self.logger.info(f"Restored model for task: {task_name}")

    def restore_model(self, task_name, task_config):
        model_dir = task_config["model_dir"]
        model_cfg = task_config["model_cfg"]
        cfg = CfgNode(model_cfg)
        # TODO: need to judge whether model_dir exists or not
        # if not, use the original config
        cfg.MODEL.WEIGHTS = model_dir

        self.logger.info(f"Model config {cfg.MODEL}")

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

    # save the task
    def save(self, task_name):
        # save model and images first, then save the config
        # this is for fault tolerance
        self.save_model(task_name)
        self.save_image(task_name)
        self.save_config(task_name)

    def save_config(self, task_name):
        task_config = json.dumps(self.task_configs[task_name])
        self.redis.set(task_name, task_config)

    def save_model(self, task_name):
        task_config = self.task_configs[task_name]
        task_model = self.task_models[task_name]

        # use ray.get to block, ensure return when model is successfully saved
        ray.get(task_model.save.remote(task_config["model_dir"]))

        # TODO: save to a folder, with the functionality of lastest checkpoint

    def save_image(self, task_name):
        task_config = self.task_configs[task_name]
        task_images = self.task_images[task_name]

        torch.save(task_images, task_config["image_dir"])

    def listen(self):
        self.setup_handler_for_channel(self.model_register_channel, self.register_handler)
        self.setup_handler_for_channel(self.model_status_channel, self.status_handler)

    def register_handler(self, register_message):
        # decode message (do we need to put all the decode process to another file to keep here cleaner?)
        register_message = json.loads(register_message["data"])
        task_type = "box2d"
        if "taskType" in register_message:
            task_type = register_message["taskType"]
        project_name = register_message["projectName"]
        task_id = register_message["taskId"]
        item_list = register_message["items"]

        self.register_task(task_type, project_name, task_id, item_list)

        self.logger.info(f"Set up model inference for {project_name}: {task_id}.")

    def status_handler(self, status_message):
        # decode message
        status_message = json.loads(status_message["data"])
        project_name = status_message["projectName"]
        task_id = status_message["taskId"]
        active = status_message["active"]

        task_name = f"{project_name}_{task_id}"

        model_notify_channel = self.model_notify_channel % task_name
        # if the corresponding task does not exist, return INVALID
        if task_name not in self.task_configs:
            self.redis.publish(model_notify_channel, ModelStatus.INVALID.value)
            return

        model = self.task_models[task_name]

        if self.task_configs[task_name]["active"] != active:
            # change the status according to the message, and save the change
            self.task_configs[task_name]["active"] = active
            self.save_config(task_name)

            if active:
                model.activate.remote()
                self.logger.info(f"{task_name} reset to active.")
            else:
                model.idle.remote()
                self.logger.info(f"{task_name} recevied no action for a period. Set to idle.")

        if active:
            self.redis.publish(model_notify_channel, ModelStatus.READY.value)
        else:
            self.redis.publish(model_notify_channel, ModelStatus.IDLE.value)

    def request_handler(self, request_message):
        # decode request message
        request_message = json.loads(request_message["data"])
        project_name = request_message["projectName"]
        task_id = request_message["taskId"]
        items = request_message["items"]
        item_indices = request_message["itemIndices"]
        action_packet_id = request_message["actionPacketId"]
        request_type = request_message["type"]

        self.logger.info("Handling request!", request_message)

        if request_type == QueryConsts.QUERY_TYPES["inference"]:
            request_data = {
                "name": f"{project_name}_{task_id}",
                "items": items,
                "item_indices": item_indices,
                "action_packet_id": action_packet_id,
                "request_type": request_type,
            }

            task_name = f"{project_name}_{task_id}"

            task_config = self.task_configs[task_name]
            task_images = self.task_images[task_name]
            inputs = [task_images[item["url"]] for item in items][0]

            # for polygon annotation
            if task_config["task_type"] == "polygon2d":
                inputs["poly"] = None
                inputs["bbox"] = None
                if items[0]["labels"][0]["box2d"] is not None:
                    box = items[0]["labels"][0]["box2d"]
                    inputs.update({"bbox": [box["x1"], box["y1"], box["x2"], box["y2"]]})

            # for box3d annotation
            if task_config["task_type"] == "box3d":
                if items[0]["intrinsics"] is not None:
                    intrinsics = items[0]["intrinsics"]
                    inputs.update({"intrinsics": torch.tensor(intrinsics)})

            model = self.task_models[task_name]
            model.remote(inputs, request_data, request_type)

    def register_task(self, task_type, project_name, task_id, item_list):
        # register task

        task_name = f"{project_name}_{task_id}"
        self.logger.info(f"Handling register! {task_name}")
        model_notify_channel = self.model_notify_channel % task_name

        # if current task name is in configs, just check whether it is active no
        if task_name in self.task_configs:
            if not self.task_configs[task_name]["active"]:
                self.task_configs[task_name]["active"] = True
                self.save_config(task_name)

                self.task_models[task_name].activate.remote()

            self.redis.publish(model_notify_channel, ModelStatus.READY.value)
            return
        # if current task name is not in configs, check redis
        elif task_name in self.redis_task_names:
            self.redis.publish(model_notify_channel, ModelStatus.LOADING.value)

            self.task_configs[task_name] = self.get_task_config(task_name)
            self.task_configs[task_name]["active"] = True

            self.save_config(task_name)

            self.restore_model(task_name, self.task_configs[task_name])
            self.restore_image(task_name, self.task_configs[task_name])
        # if it is a new task, register it
        else:
            self.redis.publish(model_notify_channel, ModelStatus.LOADING.value)

            model_registry_config = self.model_registry_config[task_type]

            model_name = model_registry_config["models"][model_registry_config["defaults"]["model"]]
            deploy_config = model_registry_config["defaults"]["deploy_config"]

            self.initialize(task_type, task_name, model_name, deploy_config, item_list)

            self.save(task_name)
            self.redis.sadd("ModelServerTasks", task_name)

        model_request_channel = self.model_request_channel % task_name
        self.setup_handler_for_channel(model_request_channel, self.request_handler)

        self.redis.publish(model_notify_channel, ModelStatus.READY.value)

    def initialize(self, task_type, task_name, model_name, deploy_config, item_list):
        cfg = get_cfg()
        add_general_config(cfg)
        if task_type == "box2d":
            cfg.merge_from_file(model_zoo.get_config_file(model_name))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        elif task_type == "polygon2d":
            cfg.TASK_TYPE = "polygon2d"
            add_polyrnnpp_config(cfg)
            cfg.merge_from_file(model_name)
        elif task_type == "box3d":
            cfg.TASK_TYPE = "box3d"
            add_dd3d_config(cfg)
            cfg.merge_from_file(model_name)
        else:
            raise NotImplementedError

        num_replicas = deploy_config["num_replicas"]
        self.logger.info(f"Model config {cfg.MODEL}")
        RayModel.options(name=task_name, num_replicas=num_replicas).deploy(
            cfg,
            self.logger,
        )

        deploy_model = serve.get_deployment(task_name).get_handle()
        self.task_configs[task_name] = {
            "task_type": task_type,
            "task_name": task_name,
            "deploy_config": deploy_config,
            "item_list": item_list,
            "model_cfg": cfg,
            "model_dir": task_name + ".pth",
            "image_dir": task_name + "_image.pkl",
            "active": True,
        }
        self.task_models[task_name] = deploy_model

        # TODO: change this to multi-processing
        self.task_images[task_name] = ray.get(deploy_model.load_inputs.remote(item_list))

    def close(self, tash_name):
        self.threads[tash_name].stop()


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
    server_config = {"redis_host": RedisConsts.REDIS_HOST, "redis_port": RedisConsts.REDIS_PORT}
    model_registry_config = {
        "tasks": ["box2d", "polygon2d", "Mask"],
        "box2d": {
            "models": {
                "R50-FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "R101-FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            },
            "defaults": {
                "model": "R50-FPN",
                "deploy_config": {
                    "bs_infer": 1,
                    "bs_train": 1,
                    "batch_wait_time": 1,  # second
                    # above are currently useless
                    "num_replicas": 1,
                },
            },
        },
        "polygon2d": {
            "models": {"POLYRNN-PP": "scalabel/automatic/model_repo/configs/polyrnn_pp/polyrnn_pp.yaml"},
            "defaults": {"model": "POLYRNN-PP", "deploy_config": {"num_replicas": 1}},
        },
        "box3d": {
            "models": {"DD3D": "scalabel/automatic/model_repo/configs/dd3d/dd3d.yaml"},
            "defaults": {"model": "DD3D", "deploy_config": {"num_replicas": 1}},
        },
    }

    # create scheduler
    scheduler = RayModelServerScheduler(server_config, model_registry_config, logger)
    scheduler.listen()


if __name__ == "__main__":
    launch()
