# The synscapes evaluation script expects one json annotation file per image with the format:
# {
#     "objects": [
#         {
#             "2d": {
#                 "modal": [xmin, ymin, w, h],
#                 "amodal": [xmin, ymin, w, h]
#             },
#             "3d": {
#                 "center": [x, y, z],
#                 "dimensions": [length, width, height],
#                 "rotation": [q1, q2, q3, q4],
#             },
#             "label": str,
#             "score": float
#         }
#     ]
# }


from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti


def kitti_label_to_synscapes(label):
    conversion = {
        0: "car",  # Car
        1: "car",  # Van
        2: "truck",  # Truck
        3: "person",  # Pedestrian
        4: "person",  # Person_sitting
        5: "rider",  # Cyclist
        6: None,  # Tram
        7: None,  # Misc
        8: None,  # DontCare
    }
    return conversion[label]


def dd3d_to_cityscapes(instances):
    objects = []

    for i in range(len(instances.pred_boxes)):
        box2d = instances.pred_boxes[i].tensor[0].cpu().numpy()
        box2d = [float(x) for x in box2d]
        box3d = instances.pred_boxes3d[i]
        kitti_box3d = convert_3d_box_to_kitti(box3d)
        kitti_box3d = [float(x) for x in kitti_box3d]
        score = instances.scores_3d[i].item()
        kitti_label = instances.pred_classes[i].item()
        label = kitti_label_to_synscapes(kitti_label)
        if label is None:
            continue

        x1, y1, x2, y2 = box2d
        xmin, ymin, w2d, h2d = x1, y1, x2 - x1, y2 - y1

        w, l, h, x, y, z = kitti_box3d[:6]
        qw, qx, qy, qz = kitti_box3d[:4]

        objects.append(
            {
                "2d": {"amodal": [xmin, ymin, w2d, h2d]},
                "3d": {
                    "center": [x, y, z],
                    "dimensions": [l, w, h],
                    "rotation": [qw, qx, qy, qz],
                },
                "label": label,
                "score": score,
            }
        )

    return {"objects": objects}
