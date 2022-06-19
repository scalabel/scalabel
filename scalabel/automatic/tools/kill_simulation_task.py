import os
import redis
import json

import scalabel.automatic.consts.redis_consts as RedisConsts


def kill_task():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    state_path = "{}/../testcases/test_task_state.json".format(cur_dir)
    with open(state_path, "r") as fp:
        kill_message = json.load(fp)
        del kill_message["items"]
        kill_message = json.dumps(kill_message)

    redisClient = redis.Redis(host=RedisConsts.REDIS_HOST, port=RedisConsts.REDIS_PORT)
    redisClient.publish(RedisConsts.REDIS_CHANNELS["modelKill"], kill_message)


if __name__ == "__main__":
    kill_task()