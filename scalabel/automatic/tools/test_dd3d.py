import torch
from torch import nn
import numpy as np
import skimage.transform as transform
from collections import OrderedDict
from scalabel.automatic.model_repo.dd3d.structures.boxes3d import Boxes3D
from scalabel.automatic.tools.dd3d_to_synscapes import dd3d_to_cityscapes, kitti_label_to_synscapes
from scalabel.automatic.tools.synscapes_to_cityscapes import synscapes_to_cityscapes
import seaborn as sns
import cv2
import json

import ray
from ray import serve

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog

from scalabel.automatic.model_repo import add_general_config, add_dd3d_config
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti
from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import draw_boxes3d_cam
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


def predict(finetuned=False, samples=range(1, 100)):
    cfg = get_cfg()
    add_general_config(cfg)
    add_dd3d_config(cfg)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "../model_repo/configs/dd3d/dd3d.yaml")
    print("file_path:", file_path)
    cfg.merge_from_file(file_path)

    if finetuned:
        cfg.MODEL.CKPT = "/scratch/egroenewald/code/dd3d/pretrained_models/model_finetuned_nms_05.pth"
        cfg.MODEL.WEIGHTS = "/scratch/egroenewald/code/dd3d/pretrained_models/model_finetuned_nms_05.pth"

    print("nms", cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH)
    cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH = 0.5
    print("nms", cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH)

    cfg.TASK_TYPE = "box3d"
    cfg.MODEL.WEIGHTS = os.path.join(dir_path, "../../..", cfg.MODEL.WEIGHTS)
    cfg.MODEL.CKPT = os.path.join(dir_path, "../../..", cfg.MODEL.CKPT)

    print("ckpt", cfg.MODEL.CKPT)
    print("weights", cfg.MODEL)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_to_disk=True)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.cuda().eval()

    # sample_id = 1001
    for sample_id in samples:
        print("sample_id:", sample_id)
        synscapes_path = "/scratch-second/egroenewald/synscapes/Synscapes"
        image_path = os.path.join(synscapes_path, f"img/rgb/{sample_id}.png")
        meta_path = os.path.join(synscapes_path, f"meta/{sample_id}.json")

        img = np.array(Image.open(image_path))
        image_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

        intrinsics = (
            torch.tensor([[1590.83437, 0.0000, 771.31406], [0.0000, 1592.79032, 360.79945], [0.0000, 0.0000, 1.0000]])
            .cpu()
            .numpy()
        )
        batched_inputs = [
            {
                "image": image_tensor,
                "intrinsics": torch.tensor(intrinsics.tolist()),
            }
        ]

        # model.eval()
        with torch.no_grad():
            output = model(batched_inputs)
        instances = output[0]["instances"]
        # result = [convert_3d_box_to_kitti(box) for box in boxes3d]

        for i in range(len(instances.pred_boxes)):
            box = instances.pred_boxes[i]
            box3d = instances.pred_boxes3d[i]
            kitti_box3d = convert_3d_box_to_kitti(box3d)
            pred_class = instances.pred_classes[i]
            # print("box,box3d,kitti_box3d,class", box, box3d.vectorize().cpu().numpy(), kitti_box3d, pred_class)
            print("box,class", box.tensor.cpu().numpy(), pred_class.cpu().numpy())
            print("box3d,class", box3d.vectorize().cpu().numpy(), pred_class.cpu().numpy())
            print("kitti,class", list(kitti_box3d), pred_class.cpu().numpy())

        kitti_boxes = [[float(a) for a in list(convert_3d_box_to_kitti(box3d))] for box3d in instances.pred_boxes3d]
        print("kitti boxes", json.dumps([kitti_boxes]))
        # Visualise 3d boxes
        metadata = MetadataCatalog.get("kitti_3d")
        metadata.thing_classes = VALID_CLASS_NAMES
        metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]
        metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
        image_np = image_tensor.cpu().numpy().astype(np.uint8)
        image_np = image_np.reshape((image_np.shape[1], image_np.shape[2], 3))
        image_cv2 = cv2.imread(image_path)
        print("image shape:", image_cv2.shape)

        boxes3d = instances.pred_boxes3d
        classes = instances.pred_classes

        vis_image = draw_boxes3d_cam(image_cv2, boxes3d, classes, metadata)

        out_img_dir = "base_imgs"
        out_label_dir = "base_labels"
        if finetuned:
            out_img_dir = "finetune_imgs"
            out_label_dir = "finetune_labels"

        cv2.imwrite(f"{out_img_dir}/vis_image_synscapes_{sample_id}.png", vis_image)

        synscapes_labels = dd3d_to_cityscapes(instances)
        with open(f"{out_label_dir}/synscapes_preds_{sample_id}.json", "w") as file:
            json.dump(synscapes_labels, file)

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        get_labels = synscapes_to_cityscapes(metadata)

        with open(f"groundtruth_labels/synscapes_gt_{sample_id}.json", "w") as file:
            json.dump(get_labels, file)


if __name__ == "__main__":
    # samples = (63, 596, 736, 627, 630, 673, 869, 529, 901, 766)
    samples = range(1, 100)
    predict(samples=samples)
    predict(finetuned=True, samples=samples)
