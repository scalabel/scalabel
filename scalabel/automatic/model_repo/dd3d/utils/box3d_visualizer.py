# Copyright 2021 Toyota Research Institute.  All rights reserved.
import json
import os
from collections import OrderedDict, defaultdict
from typing import Dict

import cv2
import numpy as np
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as d2_utils
from detectron2.utils.visualizer import VisImage, _create_text_labels

from ..structures.boxes3d import Boxes3D, GenericBoxes3D
from ..structures.pose import Pose
from ..utils.geometry import project_points3d
from ..utils.visualization import change_color_brightness, draw_text, fill_color_polygon

# from ..visualizers.bev import BEVImage
# from ..visualizers.d2_visualizer import create_instances

DARK_YELLOW = (246, 190, 0)
BBOX3D_PREDICTION_FILE = "bbox3d_predictions.json"


def pretty_render_3d_box(
    box3d,
    image,
    # camera,
    K,
    *,
    class_id=None,
    class_names=None,
    score=None,
    line_thickness=3,
    color=None,
    render_label=True,
):
    """Render the bounding box on the image. NOTE: CV2 renders in place.

    Parameters
    ----------
    box3d: GenericBoxes3D

    image: np.uint8 array
        Image (H, W, C) to render the bounding box onto. We assume the input image is in *RGB* format

    K: np.ndarray
        Camera used to render the bounding box.

    line_thickness: int, default: 1
        Thickness of bounding box lines.

    font_scale: float, default: 0.5
        Font scale used in text labels.

    draw_axes: bool, default: False
        Whether or not to draw axes at centroid of box.
        Note: Bounding box pose is oriented such that x-forward, y-left, z-up.
        This corresponds to L (length) along x, W (width) along y, and H
        (height) along z.

    draw_text: bool, default: False
        If True, renders class name on box.
    """
    if not isinstance(box3d, GenericBoxes3D):
        raise ValueError(f"`box3d` must be a type of `Genericboxes3D`: {type(box3d).__str__()}")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("`image` must be a 3-channel uint8 numpy array")

    points2d = project_points3d(box3d.corners[0].cpu().numpy(), K)
    corners = points2d.T

    # Draw the sides (first)
    for i in range(4):
        cv2.line(
            image,
            (int(corners.T[i][0]), int(corners.T[i][1])),
            (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
            color,
            thickness=line_thickness,
        )
    # Draw front (in red) and back (blue) face.
    cv2.polylines(image, [corners.T[:4].astype(np.int32)], True, color, thickness=line_thickness)
    cv2.polylines(image, [corners.T[4:].astype(np.int32)], True, color, thickness=line_thickness)

    front_face_as_polygon = corners.T[:4].ravel().astype(int).tolist()
    fill_color_polygon(image, front_face_as_polygon, color, alpha=0.5)

    V = VisImage(img=image)

    if render_label:
        # Render text label. Mostly copied from Visualizer.overlay_instances()
        label = _create_text_labels([class_id], [score] if score is not None else None, class_names)[0]
        # bottom-right corner
        text_pos = tuple([points2d[:, 0].min(), points2d[:, 1].max()])
        horiz_align = "left"
        lighter_color = change_color_brightness(tuple([c / 255.0 for c in color]), brightness_factor=0.8)
        H, W = image.shape[:2]
        default_font_size = max(np.sqrt(H * W) // 90, 10)
        height_ratio = (points2d[:, 1].max() - points2d[:, 1].min()) / np.sqrt(H * W)
        font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * default_font_size

        draw_text(V.ax, label, text_pos, font_size=font_size, color=lighter_color, horizontal_alignment=horiz_align)

    image = V.get_image()
    return image


def draw_boxes3d_cam(img, boxes3d, class_ids, metadata, intrinsics=None, scores=None, render_labels=True):
    """
    Parameters
    ----------
    img: np.ndarray
        RGB image. (H, W, 3)

    boxes3d: tridet.structures.Boxes3D or np.ndarray
        If np.ndarray, it must be a length-10 vector.
    """
    if len(boxes3d) == 0:
        return img.copy()

    assert len(class_ids) == len(boxes3d), "Number of class IDs must be same with number of 3D boxes."

    if isinstance(boxes3d, Boxes3D):
        intrinsics = torch.unique(torch.inverse(boxes3d.inv_intrinsics), dim=0)
        assert len(intrinsics) == 1, "Input 3D boxes must share intrinsics."
        _intrinsics = intrinsics[0]

        if intrinsics is not None:
            assert torch.allclose(_intrinsics, torch.as_tensor(intrinsics).reshape(3, 3))
        intrinsics = _intrinsics
        K = intrinsics.detach().cpu().numpy().copy()
    else:
        assert intrinsics is not None
        K = np.float32(intrinsics).reshape(3, 3)

    class_names = [metadata.contiguous_id_to_name[class_id] for class_id in range(len(metadata.thing_classes))]

    viz_image = img.copy()
    for k, (class_id, box3d) in enumerate(zip(class_ids, boxes3d)):
        if isinstance(box3d, GenericBoxes3D):
            box3d = box3d.vectorize()[0]
        if isinstance(box3d, torch.Tensor):
            box3d = box3d.detach().cpu().numpy()
        clr = metadata.thing_colors[class_id]
        box3d = GenericBoxes3D.from_vectors([box3d])
        if scores is not None:
            score = scores[k]
        else:
            score = None
        viz_image = pretty_render_3d_box(
            box3d,
            viz_image,
            # camera,
            K,
            class_id=class_id,
            color=clr,
            class_names=class_names,
            score=score,
            line_thickness=2,
            render_label=render_labels,
        )

    return viz_image


def draw_boxes3d_bev(
    boxes3d,
    extrinsics,
    class_ids,
    image_width,
    intrinsics=None,
    color=(0, 255, 0),
    metadata=None,
    bev=None,
    point_cloud=None,
    point_cloud_extrnsics=None,
):
    if bev is None:
        bev = BEVImage(
            metric_width=160,
            metric_height=160,
            background_clr=(255, 255, 255),
        )

    if len(boxes3d) == 0:
        return bev.data, bev

    assert len(class_ids) == len(
        boxes3d
    ), f"Number of class IDs must be same with number of 3D boxes: {len(class_ids)}, {boxes3d}"

    if intrinsics is not None:
        if isinstance(intrinsics, torch.Tensor):
            K = intrinsics.detach().cpu().numpy().copy()
        else:
            K = np.array(intrinsics).reshape(3, 3)
    if isinstance(extrinsics, np.ndarray):
        extrinsics = Pose(wxyz=np.array(extrinsics[:4]), tvec=np.array(extrinsics[4:]))
    elif isinstance(extrinsics, Dict):
        extrinsics = Pose(wxyz=np.array(extrinsics["wxyz"]), tvec=np.array(extrinsics["tvec"]))

    # visuaslize frustum.
    if intrinsics is not None:
        bev.render_camera_frustrum(
            intrinsics=K, extrinsics=extrinsics, width=image_width, line_thickness=2, color=DARK_YELLOW
        )

    if point_cloud is not None:
        assert point_cloud_extrnsics is not None, "point-cloud extrinsics is missing."
        pc_extrinsics = Pose(wxyz=np.array(point_cloud_extrnsics[:4]), tvec=np.array(point_cloud_extrnsics[4:]))
        bev.render_point_cloud(point_cloud, pc_extrinsics)

    if metadata is not None:
        colors = [metadata.thing_colors[class_id] for class_id in class_ids]
    else:
        colors = [color] * len(boxes3d)

    bev.render_bounding_box_3d(boxes3d=boxes3d, extrinsics=extrinsics, colors=colors, line_thickness=3)

    bev_vis = bev.data

    return bev_vis, bev


def bev_frustum_crop(bev_vis):
    # hacky frustum cropping
    yy, xx = np.all(
        bev_vis == np.array(DARK_YELLOW).reshape(1, 1, 3), axis=2
    ).nonzero()  # pylint: disable=too-many-function-args
    if len(xx) == 0:
        return bev_vis

    x1, y1, x2, y2 = [min(xx), min(yy), max(xx), max(yy)]
    bev_vis = bev_vis[y1:y2, x1:x2, :]

    return bev_vis


class Box3DPredictionVisualizer:
    def __init__(self, cfg, dataset_name, inference_output_dir):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._input_format = cfg.INPUT.FORMAT
        self._scale = cfg.VIS.BOX3D.PREDICTIONS.SCALE
        self._render_labels = cfg.VIS.BOX3D.PREDICTIONS.RENDER_LABELS
        self._min_depth_center = cfg.VIS.BOX3D.PREDICTIONS.MIN_DEPTH_CENTER

        dataset_dicts = DatasetCatalog.get(dataset_name)

        with open(os.path.join(inference_output_dir, BBOX3D_PREDICTION_FILE), "r") as f:
            box3d_predictions = json.load(f)

        pred_instances_by_image = defaultdict(list)
        for p in box3d_predictions:
            # 'p' is key'ed by 'image_id'.
            image_id = p["image_id"]
            pred_instances_by_image[image_id].append(p)

        det3d_threshold = cfg.VIS.BOX3D.PREDICTIONS.THRESHOLD

        # This handles images with no predictions.
        for dataset_dict in dataset_dicts:
            image_id = dataset_dict["image_id"]
            img_shape = (dataset_dict["height"], dataset_dict["width"])
            pred_instances_by_image[image_id] = create_instances(
                pred_instances_by_image[image_id], img_shape, det3d_threshold, self._metadata, score_key="score_3d"
            )
        self.pred_instances_by_image = pred_instances_by_image

    def visualize(self, x):
        """

        Parameters
        ----------
        x: Dict
            One 'dataset_dict'.

        Returns
        -------
        viz_images: Dict[np.array]
            Visualizations as RGB images.
        """
        # Load image.
        img = d2_utils.read_image(x["file_name"], format=self._input_format)
        img = d2_utils.convert_image_to_rgb(img, self._input_format)

        viz_images = OrderedDict()

        # GT 3D boxes
        gt_boxes3d = [np.float32(ann["bbox3d"]) for ann in x["annotations"]]
        gt_class_ids = [ann["category_id"] for ann in x["annotations"]]
        viz_image = draw_boxes3d_cam(
            img.copy(),
            gt_boxes3d,
            gt_class_ids,
            self._metadata,
            x["intrinsics"],
            scores=None,
            render_labels=self._render_labels,
        )
        viz_images["viz_gt_boxes3d_cam"] = viz_image

        # Predicted 3D boxes.
        pred_instances = self.pred_instances_by_image[x["image_id"]]
        pred_boxes3d = pred_instances.pred_boxes3d
        if self._min_depth_center > 0.0:
            pred_boxes3d = pred_boxes3d[pred_boxes3d[:, 6] > self._min_depth_center]

        pred_class_ids = pred_instances.pred_classes
        scores = pred_instances.scores
        viz_image = draw_boxes3d_cam(
            img.copy(),
            pred_boxes3d,
            pred_class_ids,
            self._metadata,
            x["intrinsics"],
            scores=scores,
            render_labels=self._render_labels,
        )
        viz_images["viz_pred_boxes3d_cam"] = viz_image

        # GT + Pred 3D boxes on BEV.
        viz_image, bev_obj = draw_boxes3d_bev(
            GenericBoxes3D.from_vectors(gt_boxes3d),
            intrinsics=x["intrinsics"],
            extrinsics=x["extrinsics"],
            class_ids=gt_class_ids,
            image_width=img.shape[1],
            metadata=None,
            color=(0, 255, 0),
        )
        viz_image, bev_obj = draw_boxes3d_bev(
            GenericBoxes3D.from_vectors(pred_boxes3d.cpu().numpy()),
            intrinsics=x["intrinsics"],
            extrinsics=x["extrinsics"],
            class_ids=pred_class_ids,
            image_width=img.shape[1],
            metadata=None,
            color=(0, 0, 255),
            bev=bev_obj,
        )

        # By default, ["forward" of body] == ["right" of BEV image].
        # Change it to ["up" of BEV image] by rotating.
        viz_image = cv2.rotate(viz_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Crop the BEV image to show only frustum.
        viz_image = bev_frustum_crop(viz_image)

        viz_images["viz_boxes3d_bev"] = viz_image

        return viz_images


class Box3DDataloaderVisualizer:
    def __init__(self, cfg, dataset_name):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._input_format = cfg.INPUT.FORMAT
        self._scale = cfg.VIS.BOX3D.DATALOADER.SCALE
        self._render_labels = cfg.VIS.BOX3D.DATALOADER.RENDER_LABELS

    def visualize(self, x):
        # Assumption: dataloader produce widthHW images.
        img = d2_utils.convert_image_to_rgb(x["image"].permute(1, 2, 0), self._input_format)

        boxes3d = x["instances"].gt_boxes3d
        class_ids = x["instances"].gt_classes

        viz_images = OrderedDict()

        viz_image = draw_boxes3d_cam(
            img, boxes3d, class_ids, self._metadata, x["intrinsics"], scores=None, render_labels=self._render_labels
        )
        if self._scale != 1.0:
            viz_image = cv2.resize(viz_image, fx=self._scale, fy=self._scale)
        viz_images["viz_gt_boxes3d_cam"] = viz_image

        viz_image, _ = draw_boxes3d_bev(
            boxes3d,
            intrinsics=x["intrinsics"],
            extrinsics=x["extrinsics"],
            class_ids=class_ids,
            image_width=x["image"].shape[2],
            metadata=self._metadata,
            color=(0, 255, 0),
        )
        # By default, ["forward" of body] == ["right" of BEV image].
        # Change it to ["up" of BEV image] by rotating.
        viz_image = cv2.rotate(viz_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Crop the BEV image to show only frustum.
        viz_image = bev_frustum_crop(viz_image)

        # boxes3d, intrinsics, extrinsics, class_ids, image_width, color=(0, 255, 0), metadata=None, bev=None
        viz_images["viz_gt_boxes3d_bev"] = viz_image

        return viz_images
