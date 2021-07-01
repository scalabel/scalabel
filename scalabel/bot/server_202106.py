from typing import List

import os
import argparse
import logging
import redis
import json

from scalabel.bot.model import Predictor


def launch() -> None:
    """Launch processes."""
    log_f = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_f)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Launch the server on one machine."
    )
    parser.add_argument(
        "--host", dest="host", help="server hostname", default="127.0.0.1"
    )
    parser.add_argument(
        "--port", dest="port", help="server port", default=6379
    )
    parser.add_argument(
        "--cfg-path", type=str, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    parser.add_argument(
        "--request-channel", help="redis request channel", type=str, default="modelRequest"
    )
    parser.add_argument(
        "--response-channel", help="redis response channel", type=str, default="modelResponse"
    )
    args = parser.parse_args()

    # create model
    model = Predictor(args.cfg_path)

    logger.info("Model server launched.")
    # connect to redis server
    r = redis.Redis(host=args.host, port=args.port)
    # subscribe to the model request channel
    model_request_subscriber = r.pubsub(ignore_subscribe_messages=True)
    model_request_subscriber.subscribe(args.request_channel)
    for message in model_request_subscriber.listen():
        items, item_indices, action_packet_id = json.loads(message["data"])
        results = model(items)

        pred_boxes: List[List[float]] = []
        for box in results[0]["instances"].pred_boxes:
            box = box.cpu().numpy()
            pred_boxes.append(box.tolist())

        r.publish(args.response_channel, json.dumps([pred_boxes, item_indices, action_packet_id]))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # mp.set_start_method('spawn')
    launch()
