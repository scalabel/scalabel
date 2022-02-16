# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import copy

import numpy as np
import torch
from fvcore.transforms import NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import RandomFlip, ResizeShortestEdge, ResizeTransform, apply_augmentations
from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances
from detectron2.utils.comm import get_world_size

from ..layers import bev_nms
from .core import DD3D
from ..structures.boxes3d import Boxes3D

__all__ = ["DatasetMapperTTA", "DD3DWithTTA"]


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.flip = cfg.TEST.AUG.FLIP
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image

            if "intrinsics" in dic:
                intrinsics = dic["intrinsics"].cpu().numpy().astype(np.float32)
                intrinsics = tfms.apply_intrinsics(intrinsics)
                dic["intrinsics"] = torch.as_tensor(intrinsics)
                dic["inv_intrinsics"] = torch.as_tensor(np.linalg.inv(intrinsics))
            ret.append(dic)

        return ret


class DD3DWithTTA(nn.Module):
    """
    A GeneralizedRCNN with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`GeneralizedRCNN.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(model, DD3D), "DD3DwithTTA only supports on DD3D. Got a model of type {}".format(type(model))
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

    def _batch_inference(self, batched_inputs):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        outputs = []
        inputs = []
        for idx, input in enumerate(batched_inputs):
            inputs.append(input)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                # This runs NMS per each augmented image.
                outputs.extend([res["instances"] for res in self.model(inputs)])
                inputs = []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`DD3D`
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

        return [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

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

            if not self.model.only_box2d and self.model.do_bev_nms:
                # Bird-eye-view NMS.
                keep = bev_nms(
                    merged_instances.pred_boxes3d,
                    merged_instances.scores_3d,
                    self.model.bev_nms_iou_thresh,
                    class_idxs=merged_instances.pred_classes,
                    class_agnostic=False,
                )
                merged_instances = merged_instances[keep]

        return {"instances": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_instances(self, augmented_inputs, tfms, orig_shape):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_boxes3d = []
        all_scores = []
        all_scores_3d = []
        all_classes = []
        for input, output, tfm in zip(augmented_inputs, outputs, tfms):
            # Need to invert the transforms on boxes, to obtain results on original image
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

            all_scores.extend(output.scores)
            all_scores_3d.extend(output.scores_3d)
            all_classes.extend(output.pred_classes)

        all_boxes = Boxes.cat(all_boxes)
        all_boxes3d = Boxes3D.cat(all_boxes3d)

        all_scores = torch.cat([x.scores for x in outputs])
        all_scores_3d = torch.cat([x.scores_3d for x in outputs])
        all_classes = torch.cat([x.pred_classes for x in outputs])

        return Instances(
            image_size=orig_shape,
            pred_boxes=all_boxes,
            pred_boxes3d=all_boxes3d,
            pred_classes=all_classes,
            scores=all_scores,
            scores_3d=all_scores_3d,
        )
