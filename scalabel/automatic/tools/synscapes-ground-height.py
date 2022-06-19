import os
import json
import numpy as np


def main(sample):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    meta_path = os.path.join(dir_path, "../../../local-data/items/synscapes-sample/meta", f"{sample}.json")

    with open(meta_path) as f:
        metadata = json.load(f)

    boxes3d = metadata["instance"]["bbox3d"]

    points = []
    for key in boxes3d:
        box = boxes3d[key]
        origin = np.array(box["origin"]) + np.array([1.7, 0.1, 1.22])
        box_dist = np.linalg.norm(origin)
        if box_dist < 50:
            print(origin)
            points.append(origin)

    points = np.array(points)

    x = points - np.mean(points, axis=0)
    svd = np.linalg.svd(np.dot(x.T, x))
    normal = svd[0][:, -1]
    center = np.mean(points, axis=0)

    unit_normal = normal / np.linalg.norm(normal)

    distance_to_origin = np.abs((-center).dot(unit_normal))

    print("distance", distance_to_origin, normal, center)


if __name__ == "__main__":
    sample = 1099
    main(sample)
