import json


def kitti_metadata(filepath):
    locations = []
    ts = 1
    with open(filepath) as f:
        for line in f:
            vals = line.split(" ")
            location = {
                "latitude": vals[0],
                "longitude": vals[1],
                "timestamp": ts,
                "speed": 0.0,
                "accuracy": 10.0,
                "course": 0.0,
            }
            locations.append(location)
            ts += 1

    return {"locations": locations}


if __name__ == "__main__":
    filepath = "local-data/items/kitti-tracking/oxts/training/oxts/0001.txt"
    data = kitti_metadata(filepath)
    outpath = "locations-0001.json"
    with open(outpath, "w") as f:
        json.dump(data, f)
