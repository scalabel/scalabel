from typing import List

import os
import logging
import redis
import json
import torch.multiprocessing as mp

from scalabel.bot.model import Predictor


class ModelServerScheduler(object):
    def __init__(self, server_config, model_config, logger):
        self.server_config = server_config
        self.model_config = model_config

        self.redis = redis.Redis(host=server_config["redis_host"], port=server_config["redis_port"])

        self.model_register_channel = "modelRegister"
        self.model_request_channel = "modelRequest_%s_%s"
        self.model_response_channel = "modelResponse_%s_%s"

        self.tasks = {}

        self.threads = {}

        self.logger = logger

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
        request_message = json.loads(request_message["data"])
        project_name = request_message["projectName"]
        task_id = request_message["taskId"]
        items = request_message["items"]
        item_indices = request_message["itemIndices"]
        action_packet_id = request_message["actionPacketId"]

        model = self.tasks[f'{project_name}_{task_id}']["model"]

        results = model(items)

        pred_boxes: List[List[float]] = []
        for box in results[0]["instances"].pred_boxes:
            box = box.cpu().numpy()
            pred_boxes.append(box.tolist())

        model_response_channel = self.model_response_channel % (project_name, task_id)
        self.redis.publish(model_response_channel, json.dumps([pred_boxes, item_indices, action_packet_id]))

    def register_task(self, project_name, task_id, item_list):
        model = self.get_model(self.model_config["model_name"], item_list)
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
        model = Predictor(model_name, item_list, 8, self.logger)
        return model

    def put_model(self, model, put_policy=None):
        pass

    def close(self):
        for thread_name, thread in self.threads.items():
            thread.stop()


def launch() -> None:
    """Launch processes."""
    log_f = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_f)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create scheduler
    server_config = {
        "redis_host": "127.0.0.1",
        "redis_port": 6379
    }
    model_config = {
        "model_name": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    }

    scheduler = ModelServerScheduler(server_config, model_config, logger)
    scheduler.listen()
    logger.info("Model server launched.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    mp.set_start_method("spawn")
    launch()
