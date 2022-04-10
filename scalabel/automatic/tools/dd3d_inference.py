import torch
from torch import nn
import numpy as np
import skimage.transform as transform
from collections import OrderedDict
from scalabel.automatic.model_repo.dd3d.structures.boxes3d import Boxes3D
import seaborn as sns
import cv2
import json
from scalabel.profiling.profile import profile

from codetiming import Timer


from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog

from scalabel.automatic.model_repo import add_general_config, add_dd3d_config
from scalabel.automatic.model_repo.dd3d.utils.convert import (
    convert_3d_box_to_kitti,
)
from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import (
    draw_boxes3d_cam,
)
from scalabel.automatic.model_repo.dd3d.utils.visualization import (
    float_to_uint8_color,
)

import os
from PIL import Image
import torchvision.transforms.functional as TF

import ray
from ray import serve
from ray.serve import pipeline

VALID_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")

COLORS = [
    float_to_uint8_color(clr)
    for clr in sns.color_palette("bright", n_colors=8)
]
COLORMAP = OrderedDict(
    {
        "Car": COLORS[2],  # green
        "Pedestrian": COLORS[1],  # orange
        "Cyclist": COLORS[0],  # blue
        "Van": COLORS[6],  # pink
        "Truck": COLORS[5],  # brown
        "Person_sitting": COLORS[4],  #  purple
        "Tram": COLORS[3],  # red
        "Misc": COLORS[7],  # gray
    }
)


def createFiles(num):
    files = []
    for i in range(num):
        files.append(
            "/home/cwlroda/projects/scalabel/local-data/items/kitti/image_02/000000000{}.png".format(
                i
            )
        )
    return files


# @profile(sort_by="cumulative", strip_dirs=True)
# @serve.deployment
def run(mode: str):
    timer = Timer(name="dd3d_inference")
    cfg = get_cfg()
    add_general_config(cfg)
    add_dd3d_config(cfg)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "../model_repo/configs/dd3d/dd3d.yaml")
    # print("file_path:", file_path)
    cfg.merge_from_file(file_path)

    cfg.TASK_TYPE = "box3d"
    cfg.MODEL.WEIGHTS = os.path.join(dir_path, "../../..", cfg.MODEL.WEIGHTS)
    cfg.MODEL.CKPT = os.path.join(dir_path, "../../..", cfg.MODEL.CKPT)
    # print("weights", cfg.MODEL)

    timer.start()
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_to_disk=True)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    if mode == "cuda":
        model.cuda().eval()
    else:
        model.cpu().eval()
    print(f"Model init:")
    timer.stop()

    intrinsics = (
        torch.tensor(
            [
                [721.5377, 0.0000, 609.5593],
                [0.0000, 721.5377, 172.854],
                [0.0000, 0.0000, 1.0000],
            ]
        )
        .cpu()
        .numpy()
    )

    single_input = []
    batched_inputs = []
    num = 10
    files = createFiles(num)
    for image_path in files:
        # "/Users/elrich/code/eth/scalabel/local-data/items/kitti/tracking/training/image_02/0001/000000.png"
        # image_path = "/home/cwlroda/projects/scalabel/local-data/items/kitti/image_02/0000000001.png"
        # image_path = os.path.join(dir_path, "../../../0000000019.png")
        image = Image.open(image_path)
        image_tensor = TF.to_tensor(image)

        if single_input == []:
            single_input.append(
                {
                    "image": image_tensor,
                    "intrinsics": torch.tensor(intrinsics.tolist()),
                }
            )

        batched_inputs.append(
            {
                "image": image_tensor,
                "intrinsics": torch.tensor(intrinsics.tolist()),
            }
        )

    # model.eval()
    timer.start()
    with torch.no_grad():
        output = model(single_input)
    print(f"Single task inference:")
    timer.stop()

    timer.start()
    with torch.no_grad():
        output = model(batched_inputs)
    print(f"{num} task inference:")
    timer.stop()
    print(f"Avg inference time: {timer.last / num}")
    # print(output)
    # instance = output[0]["instances"]
    # result = [convert_3d_box_to_kitti(box) for box in boxes3d]
    # boxes3d = []

    # for i in range(len(instance.pred_boxes)):
    #     box = instance.pred_boxes[i]
    #     box3d = instance.pred_boxes3d[i]
    #     boxes3d.append(box3d.vectorize().tolist()[0])
    #     print("boxes3d:", boxes3d)
    #     kitti_box3d = convert_3d_box_to_kitti(box3d)
    #     pred_class = instance.pred_classes[i]
    # print("box,box3d,kitti_box3d,class", box, box3d.vectorize().cpu().numpy(), kitti_box3d, pred_class)
    # print("box,class", box.tensor.cpu().numpy(), pred_class.cpu().numpy())
    # print(
    #     "box3d,class",
    #     box3d.vectorize().cpu().numpy(),
    #     pred_class.cpu().numpy(),
    # )
    # print("kitti,class", list(kitti_box3d), pred_class.cpu().numpy())

    # kitti_boxes = [
    #     [float(a) for a in list(convert_3d_box_to_kitti(box3d))]
    #     for box3d in instance.pred_boxes3d
    # ]
    # print("kitti boxes", json.dumps([kitti_boxes]))
    # # Visualise 3d boxes
    # metadata = MetadataCatalog.get("kitti_3d")
    # metadata.thing_classes = VALID_CLASS_NAMES
    # metadata.thing_colors = [
    #     COLORMAP[klass] for klass in metadata.thing_classes
    # ]
    # metadata.contiguous_id_to_name = {
    #     idx: klass for idx, klass in enumerate(metadata.thing_classes)
    # }
    # image_np = image_tensor.cpu().numpy().astype(np.uint8)
    # image_np = image_np.reshape((image_np.shape[1], image_np.shape[2], 3))
    # image_cv2 = cv2.imread(image_path)
    # print("image shape:", image_cv2.shape)

    # predictions = (
    #     torch.tensor(
    #         [
    #             [
    #                 0.5195724368095398,
    #                 0.5127660632133484,
    #                 -0.4780527651309967,
    #                 0.48844754695892334,
    #                 2.5968940258026123,
    #                 0.8777044415473938,
    #                 6.749297618865967,
    #                 1.605194091796875,
    #                 3.9524614810943604,
    #                 1.3974783420562744,
    #             ],
    #             [
    #                 0.49372586607933044,
    #                 0.495210736989975,
    #                 0.49991142749786377,
    #                 -0.510969340801239,
    #                 -4.852385520935059,
    #                 1.1608902215957642,
    #                 11.252735137939453,
    #                 1.6114134788513184,
    #                 4.02348518371582,
    #                 1.4812047481536865,
    #             ],
    #             [
    #                 0.49331000447273254,
    #                 0.494437038898468,
    #                 0.5051764845848083,
    #                 -0.5069259405136108,
    #                 -4.561218738555908,
    #                 1.2230360507965088,
    #                 32.35547637939453,
    #                 1.6999049186706543,
    #                 4.089993476867676,
    #                 1.5372989177703857,
    #             ],
    #             [
    #                 0.4884650707244873,
    #                 0.4843449592590332,
    #                 0.5070516467094421,
    #                 -0.5193365812301636,
    #                 -4.813807487487793,
    #                 1.1826457977294922,
    #                 16.36737060546875,
    #                 1.5933693647384644,
    #                 3.8068439960479736,
    #                 1.5679268836975098,
    #             ],
    #         ]
    #     )
    #     .cpu()
    #     .numpy()
    # )

    # predictions = torch.Tensor(boxes3d).cpu().numpy()
    # boxes3d = Boxes3D.from_vectors(predictions, intrinsics)
    # classes = [0 for _ in boxes3d]

    # # vis_image = draw_boxes3d_cam(image_cv2, instance.pred_boxes3d, instance.pred_classes, metadata)
    # vis_image = draw_boxes3d_cam(image_cv2, boxes3d, classes, metadata)
    # cv2.imwrite("out/vis_image.png", vis_image)

    # print("num boxes:", len(result))
    # filtered = [box for box in result if box[0] != 0.0 and box[1] != 0.0 and box[2] != 0]
    # print("num filtered:", len(filtered))
    # print("filtered", filtered)
    # print("Output", result)


if __name__ == "__main__":
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     with_flops=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         "scalabel/profiling/tb"
    #     ),
    # ) as p:

    # ray.init(num_cpus=64, num_gpus=8)
    # serve.start()
    print("Running with CPU")
    run("cpu")
    print("Running with GPU")
    run("cuda")

    # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
