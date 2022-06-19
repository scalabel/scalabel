from typing import List

import os
import logging
import redis
import json
import time
import torch.multiprocessing as mp

from scalabel.automatic.models.ray_model import RayModel
import scalabel.automatic.consts.redis_consts as RedisConsts
import scalabel.automatic.consts.query_consts as QueryConsts

import ray
from ray import serve

ray.init()
serve.start(http_options={"port": 8001})


class RayModelServerScheduler(object):
    def __init__(self, server_config, model_config, logger):
        self.server_config = server_config
        self.model_config = model_config

        self.redis = redis.Redis(host=server_config["redis_host"], port=server_config["redis_port"])

        self.model_register_channel = RedisConsts.REDIS_CHANNELS["modelRegister"]
        self.model_request_channel = RedisConsts.REDIS_CHANNELS["modelRequest"]
        self.model_response_channel = RedisConsts.REDIS_CHANNELS["modelResponse"]

        self.inference_batch_size = 1
        self.train_batch_size = 1
        self.inference_request_queue = []
        self.train_request_queue = []

        self.results_queue = []

        self.tasks = {}
        self.threads = {}

        self.logger = logger
        self.verbose = False

        self.cao = time.time()
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

        project_name = register_message["projectName"]
        task_id = register_message["taskId"]
        item_list = register_message["items"]

        self.register_task(project_name, task_id, item_list)

        self.logger.info(f"Set up model inference for {project_name}: {task_id}.")

    def request_handler(self, request_message):
        self.calc_time(init=True)

        request_message = json.loads(request_message["data"])
        project_name = request_message["projectName"]
        task_id = request_message["taskId"]

        items = request_message["items"]
        item_indices = request_message["itemIndices"]
        action_packet_id = request_message["actionPacketId"]
        request_type = request_message["type"]

        if request_type == QueryConsts.QUERY_TYPES["inference"]:
            self.inference_request_queue.append({
                "items": items,
                "item_indices": item_indices,
                "action_packet_id": action_packet_id,
                "request_type": request_type
            })

            self.calc_time("save data to query list")

            if len(self.results_queue) == 0:
                self.cao = time.time()

            model = self.tasks[f'{project_name}_{task_id}']["model"]
            result_id = model.remote(items, QueryConsts.QUERY_TYPES["inference"])
            self.results_queue.append(result_id)

            if len(self.results_queue) == 1000:
                self.logger.info("I'm here!")
                ray.get(self.results_queue)
                print(time.time() - self.cao)
                self.results_queue = []


        #     if len(self.inference_request_queue) == self.inference_batch_size:
        #         model = self.tasks[f'{project_name}_{task_id}']["model"]
        #
        #         items = [self.inference_request_queue[i]["items"][0] for i in range(self.inference_batch_size)]
        #         self.calc_time("pack data")
        #
        #         results = ray.get(model.remote(items, QueryConsts.QUERY_TYPES["inference"]))
        #
        #         self.calc_time("model inference time")
        #
        #         model_response_channel = self.model_response_channel % (project_name, task_id)
        #         for i in range(self.inference_batch_size):
        #             pred_boxes: List[List[float]] = []
        #             # pred_boxes.append([1.0, 1.0, 100.0, 100.0])
        #             for box in results[i]["instances"].pred_boxes:
        #                 box = box.cpu().numpy()
        #                 pred_boxes.append(box.tolist())
        #             item_indices = self.inference_request_queue[i]["item_indices"]
        #             action_packet_id = self.inference_request_queue[i]["action_packet_id"]
        #             self.redis.publish(model_response_channel, json.dumps([pred_boxes, item_indices, action_packet_id]))
        #         self.inference_request_queue = []
        #
        #         self.calc_time("response time")
        # else:
        #     self.train_request_queue.append({
        #         "items": items,
        #         "item_indices": item_indices,
        #         "action_packet_id": action_packet_id,
        #         "request_type": request_type
        #     })
        #
        #     self.calc_time("save data to query list")
        #
        #     if len(self.train_request_queue) == self.train_batch_size:
        #         model = self.tasks[f'{project_name}_{task_id}']["model"]
        #
        #         items = [self.train_request_queue[i]["items"][0] for i in range(self.train_batch_size)]
        #         self.calc_time("pack data")
        #
        #         ray.get(model.remote(items, QueryConsts.QUERY_TYPES["training"]))
        #         self.calc_time("model training time")
        #
        #         self.train_request_queue = []
        #
        #     if len(self.train_request_queue) == 0:
        #         pass

    def register_task(self, project_name, task_id, item_list):
        task_model_config = self.model_config["OD"]
        model_name = task_model_config["defaults"]["model"]
        model = self.get_model(task_model_config["models"][model_name], item_list)
        self.put_model(model)

        self.tasks[f'{project_name}_{task_id}'] = {
            "project_name": project_name,
            "task_id": task_id,
            "model": model,
        }

        model_request_channel = self.model_request_channel % (project_name, task_id)
        model_request_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_request_subscriber.subscribe(**{model_request_channel: self.request_handler})
        thread = model_request_subscriber.run_in_thread(sleep_time=0.001)
        self.threads[model_request_channel] = thread

    def get_model(self, model_name, item_list):
        RayModel.options(name="test").deploy(model_name, item_list, 1, self.logger)
        model = serve.get_deployment("test").get_handle()
        # model = RayModel.remote(model_name, item_list, 1, self.logger)
        return model

    def put_model(self, model, put_policy=None):
        pass

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
        "task": ["OD", "Polygon", "Mask"],
        "OD": {
            "models": {
                "R50-FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "R101-FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            },
            "defaults": {
                "model": "R50-FPN",
                "bs_infer": 1,
                "bs_train": 1,
                "batch_wait_time": 1  # second
            }
        }
    }

    scheduler = RayModelServerScheduler(server_config, model_config, logger)
    scheduler.listen()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
    launch()