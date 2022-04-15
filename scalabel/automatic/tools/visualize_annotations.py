"""
Adapted from https://github.com/lzccccc/3d-bounding-box-estimation-for-autonomous-driving/blob/master/utils/visualization3Dbox.py
"""
from lib2to3.pytree import convert
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image
from scalabel.automatic.model_repo.dd3d.utils.convert import convert_3d_box_to_kitti
from scalabel.automatic.tools.scalabel_to_dd3d import scalabel_to_dd3d

from scalabel.automatic.tools.synscapes_to_dd3d import synscapes_to_dd3d


def compute_birdviewbox(box3d, shape, scale):
    w, l, _, x, _, z, rot_y, _ = box3d
    w, l, x, z = w * scale, l * scale, x * scale, z * scale

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)], [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -l / 2
    z_corners += -w / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape / 2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0, :]))


def draw_birdeyes(ax, box3d, shape, scale, gt=False):
    corners_2d = compute_birdviewbox(box3d, shape, scale)

    codes = [Path.LINETO] * corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(corners_2d, codes)
    color = "orange" if gt else "green"
    label = "ground truth" if gt else "prediction"
    p = patches.PathPatch(pth, fill=False, color=color, label=label)
    ax.add_patch(p)


def plot_bev(pred_boxes=[], gt_boxes=[], save_path=None, fov=np.pi / 2):
    shape = 900
    scale = 15
    fig, ax = plt.subplots()

    for pred_box in pred_boxes:
        new_scale = scale
        # if save_path is not None and "1083.png" in save_path:
        #     new_scale = scale * (1.22 / 1.5)
        draw_birdeyes(ax, pred_box, shape, new_scale)

    for gt_box in gt_boxes:
        draw_birdeyes(ax, gt_box, shape, scale, gt=True)

    # plot camera view range
    slope = np.tan((np.pi - fov) / 2)
    slope = 2
    x1 = np.linspace(0, shape / 2)
    x2 = np.linspace(shape / 2, shape)
    ax.plot(x1, slope * (shape / 2 - x1), ls="--", color="grey", linewidth=1, alpha=0.5)
    ax.plot(x2, slope * (x2 - shape / 2), ls="--", color="grey", linewidth=1, alpha=0.5)
    for i in range(1, 11):
        circle = plt.Circle((shape / 2, 0), i * 10 * scale, color="grey", fill=False, ls="--", linewidth=1, alpha=0.5)
        ax.add_patch(circle)
    ax.plot(shape / 2, 0, marker="+", markersize=16, markeredgecolor="red")

    # visualize bird eye view
    birdimage = np.zeros((shape, shape, 3), np.uint8)
    ax.imshow(birdimage, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        [handles[0], handles[-1]], [labels[0], labels[-1]], loc="lower right", fontsize="x-small", framealpha=0.2
    )
    for text in legend.get_texts():
        plt.setp(text, color="w")

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    samples = list(range(1080, 1086))
    for sample in samples:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        gt_path = os.path.join(dir_path, "../../../local-data/items/synscapes-sample/meta/", f"{sample}.json")
        filename = "synscapes-1080_export_2022-04-12_15-36-09.json"
        file_path = os.path.join(dir_path, "../../..", f"local-data/annotations/{filename}")
        save_path = os.path.join(dir_path, "../../..", f"local-data/annotations/bev_imgs/{sample}.png")
        with open(file_path) as f:
            file_data = json.load(f)

        labels_dict = {}
        for item in file_data["frames"]:
            key = item["name"]
            labels = item["labels"]
            labels_dict[key] = labels

        with open(gt_path) as f:
            metadata = json.load(f)

        gt_boxes = synscapes_to_dd3d(metadata)
        gt_boxes = [convert_3d_box_to_kitti(box["box3d"]) for box in gt_boxes]

        pred_labels = labels_dict[f"img_00{sample}"]
        pred_boxes = scalabel_to_dd3d(pred_labels)
        pred_boxes = [convert_3d_box_to_kitti(box) for box in pred_boxes]

        fx = 1590.83437
        resx = 1440
        fov = 2 * np.arctan(resx / (2 * fx))
        plot_bev(pred_boxes, gt_boxes, fov=fov, save_path=save_path)
