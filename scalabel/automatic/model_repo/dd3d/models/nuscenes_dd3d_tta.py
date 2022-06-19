# Copyright 2021 Toyota Research Institute.  All rights reserved.
import copy

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances
from detectron2.utils.comm import get_world_size

from ..layers import bev_nms
from .nuscenes_dd3d import NuscenesDD3D
from .postprocessing import get_group_idxs, nuscenes_sample_aggregate
from .test_time_augmentation import DatasetMapperTTA
from ..structures.boxes3d import Boxes3D


class NuscenesDD3DWithTTA(nn.Module):
    def __init__(self, cfg, model, tta_mapper=None):
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, NuscenesDD3D
        ), "NuscenesDD3DWithTTA only supports on NuscenesDD3D. Got a model of type {}".format(type(model))

        assert (
            not model.postprocess_in_inference
        ), "To use test-time augmentation, `postprocess_in_inference` must be False."
        self.cfg = cfg.copy()

        self.model = model
        self.nms_thresh = cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = cfg.TEST.IMS_PER_BATCH // get_world_size()

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`NuscenesDD3D`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.tta_mapper.image_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        instances_per_image = [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

        # ----------------------------------------------------------
        # NuScenes specific: cross-image (i.e. sample-level) BEV NMS.
        # ----------------------------------------------------------
        sample_tokens = [x["sample_token"] for x in batched_inputs]
        group_idxs = get_group_idxs(sample_tokens, self.model.num_images_per_sample)
        global_poses = [x["pose"] for x in batched_inputs]

        filtered_instances = nuscenes_sample_aggregate(
            instances_per_image,
            group_idxs,
            self.model.num_classes,
            global_poses,
            self.model.bev_nms_iou_thresh,
            max_num_dets_per_sample=self.model.max_num_dets_per_sample,
        )

        return [{"instances": instances} for instances in filtered_instances]

    def _inference_one_image(self, x):
        """
        Args:
            x (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (x["height"], x["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(x)
        merged_instances = self._get_augmented_instances(augmented_inputs, tfms, orig_shape)
        if len(merged_instances) > 0:
            if self.model.do_nms:
                # Multiclass NMS.
                keep = batched_nms(
                    merged_instances.pred_boxes.tensor,
                    merged_instances.scores_3d,
                    merged_instances.pred_classes,
                    self.nms_thresh,
                )
                merged_instances = merged_instances[keep]

            if not self.model.only_box2d and self.model.do_bev_nms > 0:
                # Bird-eye-view NMS.
                keep = bev_nms(
                    merged_instances.pred_boxes3d,
                    merged_instances.scores_3d,
                    self.model.bev_nms_iou_thresh,
                    class_idxs=merged_instances.pred_classes,
                    class_agnostic=False,
                )
                merged_instances = merged_instances[keep]

        return merged_instances

    def _get_augmented_inputs(self, x):
        augmented_inputs = self.tta_mapper(x)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_instances(self, augmented_inputs, tfms, orig_shape):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_boxes3d = []

        for input, output, tfm in zip(augmented_inputs, outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            inv_tfm = tfm.inverse()

            # 2D boxes
            pred_boxes = output.pred_boxes.tensor
            orig_pred_boxes = inv_tfm.apply_box(pred_boxes.cpu().numpy())
            orig_pred_boxes = torch.from_numpy(orig_pred_boxes).to(pred_boxes.device)
            all_boxes.append(Boxes(orig_pred_boxes))

            # 3D boxes
            pred_boxes_3d = output.pred_boxes3d
            vectorized_boxes_3d = pred_boxes_3d.vectorize().cpu().numpy()
            orig_vec_pred_boxes_3d = [inv_tfm.apply_box3d(box3d_as_vec) for box3d_as_vec in vectorized_boxes_3d]

            # intrinsics
            orig_intrinsics = inv_tfm.apply_intrinsics(input["intrinsics"].cpu().numpy())
            orig_pred_boxes_3d = Boxes3D.from_vectors(
                orig_vec_pred_boxes_3d, orig_intrinsics, device=pred_boxes_3d.device
            )
            all_boxes3d.append(orig_pred_boxes_3d)

        all_boxes = Boxes.cat(all_boxes)
        all_boxes3d = Boxes3D.cat(all_boxes3d)

        all_scores = torch.cat([x.scores for x in outputs])
        all_scores_3d = torch.cat([x.scores_3d for x in outputs])
        all_classes = torch.cat([x.pred_classes for x in outputs])

        all_attributes = torch.cat([x.pred_attributes for x in outputs])
        all_speeds = torch.cat([x.pred_speeds for x in outputs])

        return Instances(
            image_size=orig_shape,
            pred_boxes=all_boxes,
            pred_boxes3d=all_boxes3d,
            pred_classes=all_classes,
            scores=all_scores,
            scores_3d=all_scores_3d,
            pred_attributes=all_attributes,
            pred_speeds=all_speeds,
        )

    def _batch_inference(self, batched_inputs):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        outputs = []
        inputs = []
        for idx, x in enumerate(batched_inputs):
            inputs.append(x)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                # This runs NMS (box and optionally bev) per each augmented image.
                outputs.extend([res["instances"] for res in self.model(inputs)])
                inputs = []
        return outputs
