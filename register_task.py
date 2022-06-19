import redis
import json

if __name__ == "__main__":
    r = redis.Redis("127.0.0.1", 6379)
    f = open("register-data.json")
    data = json.load(f)
    f.close()
    message = json.dumps(data)
    channel = "modelRegister"
    r.publish(channel, message)
    print("Sent message!", message)
