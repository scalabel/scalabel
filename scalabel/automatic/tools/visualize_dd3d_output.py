import json
from collections import defaultdict
import shutil
from collections import OrderedDict
import seaborn as sns
import os
import cv2
from scalabel.automatic.model_repo.dd3d.structures.boxes3d import GenericBoxes3D
from scalabel.automatic.model_repo.dd3d.utils.box3d_visualizer import draw_boxes3d_cam
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti
from detectron2.data import DatasetCatalog, MetadataCatalog
from scalabel.automatic.model_repo.dd3d.utils.visualization import float_to_uint8_color
import torch

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
CALIB_DATA = [
    "P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
    "P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
    "P2: 1590.83437 0.0000 771.31406 0.0 0.0000 1592.79032 360.79945 0.0 0.000 0.0000 1.0000 0.0",
    "P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03",
    "R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01",
    "Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01",
    "Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01",
]

metadata = MetadataCatalog.get("kitti_3d")
metadata.thing_classes = VALID_CLASS_NAMES
metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]
metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}


def visualize(file_name, outpath, img_base, out_img_dir):
    with open(file_name) as f:
        data = json.load(f)

    raw_labels = defaultdict(list)
    classes = defaultdict(list)
    labels = {}

    intrinsics = (
        torch.tensor([[1590.83437, 0.0000, 771.31406], [0.0000, 1592.79032, 360.79945], [0.0000, 0.0000, 1.0000]])
        .cpu()
        .numpy()
    )

    for d in data:
        image_id = d["image_id"]
        category = d["category"]
        box3d_raw = d["bbox3d"]

        raw_labels[image_id].append(box3d_raw)
        classes[image_id].append(category)

    for image_id in raw_labels:
        class_ids = [VALID_CLASS_NAMES.index(c) for c in classes[image_id]]
        boxes3d = GenericBoxes3D.from_vectors(raw_labels[image_id])
        labels[image_id] = boxes3d
        img_path = os.path.join(img_base, f"{image_id}.png")
        image_cv2 = cv2.imread(img_path)
        vis_image = draw_boxes3d_cam(image_cv2, boxes3d, class_ids, metadata, intrinsics)
        cv2.imwrite(f"{out_img_dir}/vis_image_synscapes_{image_id}.png", vis_image)
        print(f"Saved image {image_id}")
    # for image_id in labels:
    #     # calib_outpath = os.path.join(outpath, "calib", f"{image_id}.txt")
    #     # labels_outpath = os.path.join(outpath, "label_2", f"{image_id}.txt")
    #     # img_outpath = os.path.join(outpath, "image_2", f"{image_id}.png")
    #     # img_path = os.path.join(img_base, f"{image_id}.png")

    #     with open(calib_outpath, "w") as f:
    #         for line in CALIB_DATA:
    #             f.write(line + "\n")

    #     with open(labels_outpath, "w") as f:
    #         for i, box3d in enumerate(labels[image_id]):
    #             class_label = classes[image_id][i]
    #             w, l, h, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
    #             xmin, xmax, ymin, ymax = [0, 0, 0, 0]  # todo : add actual bbox

    #             line = f"{class_label} 0.00 0 {alpha} {xmin} {ymin} {xmax} {ymax} {h} {w} {l} {x} {y} {z} {rot_y}"

    #             print("box", line)
    #             f.write(line + "\n")

    #     shutil.copyfile(img_path, img_outpath)


if __name__ == "__main__":
    # file_name = "/Users/elrich/code/eth/dd3d/outputs/gao947g6-20220324_230647/inference/step0001400/synscapes_finetune_val/bbox3d_predictions.json"
    # file_name = "/Users/elrich/code/eth/dd3d/outputs/1t3w9i4e-20220330_154237/inference/step0001400/synscapes_finetune_val/bbox3d_predictions.json"
    file_name = "/Users/elrich/code/eth/dd3d/outputs/1k18ssgc-20220407_135606/inference/step0000200/synscapes_finetune_val/bbox3d_predictions.json"
    # file_name = "/Users/elrich/code/eth/dd3d/outputs/3igeo3bl-20220407_171423/inference/step0000200/synscapes_finetune_val/bbox3d_predictions.json"
    outpath = "/Users/elrich/code/eth/simple_kitti_visualization/examples/synscapes"
    img_base = "/Users/elrich/code/eth/scalabel/local-data/items/synscapes-sample/img/rgb"
    out_img_dir = "/Users/elrich/code/eth/scalabel/local-data/items/synscapes-preds/cls_unfrozen"
    visualize(file_name, outpath, img_base, out_img_dir)
