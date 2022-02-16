import redis
import json

message_data = """
{
    "projectName": "test-bot-dd3d",
    "taskId": 0,
    "items": [
        {
            "id": "0",
            "index": 0,
            "url": "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000101.jpg",
            "intrinsics":  [[738.9661, 0.0000, 624.2830], [0.0000, 738.8547, 177.0025], [0.0000, 0.0000, 1.0000]],
            "labels": [
                {
                    "box2d": {
                        "x1": 0,
                        "x2": 100,
                        "y1": 0,
                        "y2": 100
                    }
                }
            ]
        }
    ],
    "itemIndices": [
        0
    ],
    "actionPacketId": "test",
    "type": "inference"
}
"""
if __name__ == "__main__":
    r = redis.Redis("127.0.0.1", 6379)
    # f = open("inference-data.json")
    # data = json.load(f)
    # f.close()
    data = json.loads(message_data)
    message = json.dumps(data)
    channel = "modelRequest_test-bot-dd3d_0"
    r.publish(channel, message)
    print("Sent message!", channel, message)
