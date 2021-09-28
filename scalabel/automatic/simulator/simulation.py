import os
import redis
import json
import logging
import time

import scalabel.automatic.consts.redis_consts as RedisConsts
import scalabel.automatic.consts.query_consts as QueryConsts


class Simulator(object):
    def __init__(self, logger):
        self.redis = redis.Redis(host=RedisConsts.REDIS_HOST, port=RedisConsts.REDIS_PORT)

        self.model_request_channel = RedisConsts.REDIS_CHANNELS["modelRequest"]
        self.model_response_channel = RedisConsts.REDIS_CHANNELS["modelResponse"]

        self.threads = {}

        self.state = {}

        self.logger = logger
        self.response_count = 0

    def listen(self):
        model_response_subscriber = self.redis.pubsub(ignore_subscribe_messages=True)
        model_response_subscriber.subscribe(**{self.model_response_channel: self.response_handler})
        thread = model_response_subscriber.run_in_thread(sleep_time=0.001)

        self.threads[self.model_response_channel] = thread

    def response_handler(self, response_message):
        self.response_count += 1
        if self.response_count % 100 == 99:
            self.logger.info("100 images processing time: {}".format(time.time() - self.start_time))
            self.start_time = time.time()

    def get_message_and_channel(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        state_path = "{}/../testcases/test_task_state.json".format(cur_dir)
        with open(state_path, "r") as fp:
            state = json.load(fp)
        # {
        #   items: List {[
        #       url: string
        #   ]}
        #   itemIndices: List[int]
        #   actionPacketId: str
        #   "type": str, defines later to indicate whether it is inference or training
        # }
        state["items"] = [state["items"][0]]
        state["items"][0]["url"] = state["items"][0]["urls"]["-1"]
        state["items"][0].pop("urls")
        state["itemIndices"] = [0]
        state["actionPacketId"] = "test"

        self.state = state
        self.model_request_channel = self.model_request_channel % (
            state["projectName"], state["taskId"])
        self.model_response_channel = self.model_response_channel % (
            state["projectName"], state["taskId"])

        return

    def send_inference_requests(self):
        self.start_time = time.time()
        self.state["type"] = QueryConsts.QUERY_TYPES["inference"]
        for i in range(10000):
            model_request_message = json.dumps(self.state)
            self.redis.publish(self.model_request_channel, model_request_message)

    def send_inference_and_training_requests(self):
        self.start_time = time.time()
        # This means each image will both go through the inference and training step.
        # But the training does not need to happen every step after a inference step.
        # See the two queues in server for detail.
        for i in range(1000):
            if i % 2 in (0, ):
                self.state["type"] = QueryConsts.QUERY_TYPES["inference"]
            else:
                self.state["type"] = QueryConsts.QUERY_TYPES["training"]
            model_request_message = json.dumps(self.state)
            self.redis.publish(self.model_request_channel, model_request_message)


if __name__ == "__main__":
    log_f = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_f)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # fh = logging.FileHandler('throughput_1_1.txt')
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    simulator = Simulator(logger)

    simulator.get_message_and_channel()
    simulator.listen()

    # simulator.send_inference_requests()
    simulator.send_inference_and_training_requests()