import os
import redis
import json

import scalabel.automatic.consts.redis_consts as RedisConsts


def setup_task():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    state_path = "{}/../testcases/test_task_state_polygon.json".format(cur_dir)
    with open(state_path, "r") as fp:
        register_message = json.dumps(json.load(fp))

    redisClient = redis.Redis(host=RedisConsts.REDIS_HOST, port=RedisConsts.REDIS_PORT)
    redisClient.publish(RedisConsts.REDIS_CHANNELS["modelRegister"], register_message)


if __name__ == "__main__":
    setup_task()