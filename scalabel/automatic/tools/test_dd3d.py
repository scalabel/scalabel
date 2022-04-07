import torch
from torch import nn
import numpy as np
import skimage.transform as transform
from collections import OrderedDict
from scalabel.automatic.model_repo.dd3d.augmentations.resize_transform import ResizeShortestEdge
from scalabel.automatic.model_repo.dd3d.dataset_mappers.dataset_mapper import DefaultDatasetMapper
from scalabel.automatic.model_repo.dd3d.structures.boxes3d import Boxes3D, GenericBoxes3D
from scalabel.automatic.model_repo.dd3d.utils.tasks import TaskManager
from scalabel.automatic.tools.dd3d_to_synscapes import dd3d_to_cityscapes, kitti_label_to_synscapes
from scalabel.automatic.tools.synscapes_to_cityscapes import synscapes_to_cityscapes
import seaborn as sns
import cv2
import json
from pyquaternion import Quaternion

import ray
from ray import serve

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog

from scalabel.automatic.model_repo import add_general_config, add_dd3d_config
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti
from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import draw_boxes3d_bev, draw_boxes3d_cam
from scalabel.automatic.model_repo.dd3d.utils.visualization import float_to_uint8_color
from scalabel.automatic.models.ray_model_new import RayModel

import os
from PIL import Image
import torchvision.transforms.functional as TF

VALID_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=8)]
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


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2
    )
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2
    )
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2
    )
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2
    )

    return [qw, qx, qy, qz]


def predict(finetuned=False, samples=range(1, 100)):
    cfg = get_cfg()
    add_general_config(cfg)
    add_dd3d_config(cfg)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "../model_repo/dd3d/configs/dd3d.yaml")
    cfg.merge_from_file(file_path)
    file_path = os.path.join(dir_path, "../model_repo/configs/dd3d/dd3d.yaml")
    cfg.merge_from_file(file_path)

    if finetuned:
        # cfg.MODEL.CKPT = "/scratch/egroenewald/code/dd3d/pretrained_models/model_finetuned_nms_05.pth"
        # cfg.MODEL.WEIGHTS = "/scratch/egroenewald/code/dd3d/pretrained_models/model_finetuned_nms_05.pth"
        # cfg.MODEL.CKPT = "/scratch/egroenewald/code/dd3d/outputs/1t3w9i4e-20220330_154237/model_0000699.pth"
        # cfg.MODEL.WEIGHTS = "/scratch/egroenewald/code/dd3d/outputs/1t3w9i4e-20220330_154237/model_0000699.pth"
        # Trained on new 10_2 dataset
        # cfg.MODEL.CKPT = "/scratch/egroenewald/code/dd3d/outputs/2cqf6zkg-20220406_141344/model_0001299.pth"
        # cfg.MODEL.WEIGHTS = "/scratch/egroenewald/code/dd3d/outputs/2cqf6zkg-20220406_141344/model_0001299.pth"
        # Trained on new 10_2 dataset and cls tower unfrozen
        cfg.MODEL.CKPT = "/scratch/egroenewald/code/dd3d/outputs/1k18ssgc-20220407_135606/model_0000499.pth"
        cfg.MODEL.WEIGHTS = "/scratch/egroenewald/code/dd3d/outputs/1k18ssgc-20220407_135606/model_0000499.pth"
        # Trained only on 674
        # cfg.MODEL.CKPT = "/scratch/egroenewald/code/dd3d/outputs/uuvjf98k-20220405_192229/model_0001499.pth"
        # cfg.MODEL.WEIGHTS = "/scratch/egroenewald/code/dd3d/outputs/uuvjf98k-20220405_192229/model_0001499.pth"
        # Trained with class agnostic 3d boxes
        # cfg.MODEL.CKPT = "/scratch/egroenewald/code/dd3d/outputs/javozlbm-20220406_223747/model_0000599.pth"
        # cfg.MODEL.WEIGHTS = "/scratch/egroenewald/code/dd3d/outputs/javozlbm-20220406_223747/model_0000599.pth"

    print("nms", cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH)
    cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH = 0.5
    cfg.DD3D.FCOS2D.INFERENCE.DO_NMS = True
    # cfg.DD3D.FCOS3D.CLASS_AGNOSTIC_BOX3D = True
    print(
        "nms",
        cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH,
        cfg.DD3D.FCOS2D.INFERENCE.DO_NMS,
        cfg.DD3D.FCOS3D.CLASS_AGNOSTIC_BOX3D,
    )

    cfg.TASK_TYPE = "box3d"
    cfg.MODEL.WEIGHTS = os.path.join(dir_path, "../../..", cfg.MODEL.WEIGHTS)
    cfg.MODEL.CKPT = os.path.join(dir_path, "../../..", cfg.MODEL.CKPT)

    print("ckpt", cfg.MODEL.CKPT)
    print("weights", cfg.MODEL.WEIGHTS)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_to_disk=True)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.cuda().eval()

    augs = [ResizeShortestEdge(384, 10000, "choice")]
    task_manager = TaskManager(cfg)
    is_train = False
    image_format = cfg.INPUT.FORMAT
    dataset_mapper = DefaultDatasetMapper(is_train, task_manager, augs, image_format)

    for sample_id in samples:
        print("sample_id:", sample_id)
        synscapes_path = "/scratch-second/egroenewald/synscapes/Synscapes"
        image_path = os.path.join(synscapes_path, f"img/rgb/{sample_id}.png")
        meta_path = os.path.join(synscapes_path, f"meta/{sample_id}.json")

        img = np.array(Image.open(image_path))
        image_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

        # intrinsics = (
        #     torch.tensor([[1590.83437, 0.0000, 771.31406], [0.0000, 1592.79032, 360.79945], [0.0000, 0.0000, 1.0000]])
        #     .cpu()
        #     .numpy()
        # )
        intrinsics = np.array(
            [[1590.83437, 0.0000, 771.31406], [0.0000, 1592.79032, 360.79945], [0.0000, 0.0000, 1.0000]]
        )
        batched_inputs = [
            dataset_mapper(
                {
                    "file_name": image_path,
                    "image": image_tensor,
                    # "intrinsics": torch.tensor(intrinsics.tolist()),
                    "intrinsics": intrinsics,
                }
            )
        ]

        # model.eval()
        with torch.no_grad():
            output = model(batched_inputs)
        instances = output[0]["instances"]
        # result = [convert_3d_box_to_kitti(box) for box in boxes3d]

        # for i in range(len(instances.pred_boxes)):
        #     box = instances.pred_boxes[i]
        #     box3d = instances.pred_boxes3d[i]
        #     kitti_box3d = convert_3d_box_to_kitti(box3d)
        #     pred_class = instances.pred_classes[i]
        # print("box,box3d,kitti_box3d,class", box, box3d.vectorize().cpu().numpy(), kitti_box3d, pred_class)
        # print("box,class", box.tensor.cpu().numpy(), pred_class.cpu().numpy())
        # print("box3d,class", box3d.vectorize().cpu().numpy(), pred_class.cpu().numpy())
        # print("kitti,class", list(kitti_box3d), pred_class.cpu().numpy())

        # kitti_boxes = [[float(a) for a in list(convert_3d_box_to_kitti(box3d))] for box3d in instances.pred_boxes3d]
        # print("kitti boxes", json.dumps([kitti_boxes]))
        # Visualise 3d boxes
        metadata = MetadataCatalog.get("kitti_3d")
        metadata.thing_classes = VALID_CLASS_NAMES
        metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]
        metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
        image_np = image_tensor.cpu().numpy().astype(np.uint8)
        image_np = image_np.reshape((image_np.shape[1], image_np.shape[2], 3))
        image_cv2 = cv2.imread(image_path)
        # image_cv2 = cv2.resize(image_cv2, (384 * 2, 384))
        # print("image shape:", image_cv2.shape)

        with open(meta_path, "r") as f:
            sample_meta = json.load(f)
        boxes3d = instances.pred_boxes3d
        classes = instances.pred_classes
        scores = instances.scores
        # print("inv", boxes3d.inv_intrinsics)
        # boxes3d.inv_intrinsics = torch.as_tensor(np.linalg.inv(intrinsics), device="cuda", dtype=torch.float32).expand(
        #     len(boxes3d.quat), 3, 3
        # )
        # print("inv", boxes3d.inv_intrinsics)
        # print("original intrinsics", intrinsics)

        # print car predictions and their center
        # for i in range(len(boxes3d)):
        #     pred_class = classes[i]
        #     pred_box = boxes3d[i]
        #     score = scores[i]
        #     if pred_class == 0:
        #         print("car prediction", pred_box.tvec, score)

        bev_width = 240
        extrinsics = sample_meta["camera"]["extrinsic"]
        roll = extrinsics["roll"]
        pitch = extrinsics["pitch"]
        yaw = extrinsics["yaw"]
        x = extrinsics["x"]
        y = extrinsics["y"]
        z = extrinsics["z"]
        # extrinsics = {"tvec": [x, y, z], "wxyz": get_quaternion_from_euler(roll, pitch, yaw)}
        extrinsics = {"tvec": [0, 0, 0], "wxyz": Quaternion().elements}

        vis_image = draw_boxes3d_cam(image_cv2, boxes3d.vectorize(), classes, metadata, intrinsics)
        bev_vis, bev = draw_boxes3d_bev(boxes3d, extrinsics, classes, bev_width)

        out_img_dir = "base_imgs"
        out_label_dir = "base_labels"
        if finetuned:
            out_img_dir = "finetune_imgs"
            out_label_dir = "finetune_labels"

        cv2.imwrite(f"{out_img_dir}/vis_image_synscapes_{sample_id}.png", vis_image)
        # cv2.imwrite(f"{out_img_dir}_bev/vis_image_synscapes_{sample_id}.png", bev_vis)

        synscapes_labels = dd3d_to_cityscapes(instances)
        with open(f"{out_label_dir}/synscapes_preds_{sample_id}.json", "w") as file:
            json.dump(synscapes_labels, file)

        get_labels = synscapes_to_cityscapes(sample_meta)

        with open(f"groundtruth_labels/synscapes_gt_{sample_id}.json", "w") as file:
            json.dump(get_labels, file)


if __name__ == "__main__":
    # finetune_10_2 samples
    samples = (63, 596, 736, 627, 630, 674, 869, 529, 901, 766)
    # Add one to the samples -> this is what the datamapper did
    print("samples", samples)
    # samples = range(1, 100)
    # samples = [1, 2]
    # samples = range(1080, 1100)
    # samples = [674]
    predict(finetuned=True, samples=samples)
    predict(samples=samples)
