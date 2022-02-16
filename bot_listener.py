import redis
import json


def message_handler(message):
    message_data = json.loads(message["data"])
    print("Received message!", message_data)


if __name__ == "__main__":
    r = redis.Redis("127.0.0.1", 6379)
    subscriber = r.pubsub(ignore_subscribe_messages=True)
    subscriber.subscribe(**{"modelResponse_test-bot_0": message_handler})
    subscriber.subscribe(**{"modelResponse_test-bot": message_handler})
    subscriber.subscribe(**{"modelRequest_test-bot_0": message_handler})
    subscriber.subscribe(**{"modelResponse_test-bot-dd3d_0": message_handler})
    subscriber.subscribe(**{"modelResponse_test-bot-dd3d": message_handler})
    subscriber.subscribe(**{"modelRequest_test-bot-dd3d_0": message_handler})
    thread = subscriber.run_in_thread(sleep_time=0.001)
