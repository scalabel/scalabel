# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from detectron2:
#     https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/detection_utils.py
import logging

from .color_transform import RandomBrightness, RandomContrast, RandomSaturation
from .crop_transform import RandomCrop
from .flip_transform import RandomFlip
from .resize_transform import ResizeShortestEdge

LOG = logging.getLogger(__name__)


def build_augmentation(cfg, is_train):
    """
    Changes from the original function:
        -  Move `RandomCrop` augmentation here; it's originally in dataset_mapper
            https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/dataset_mapper.py#L89
        - `RandomFlip()` is configurable. This is mostly unused for now.
        - `RandomCrop` uses expanded version of `CropTransform`, which handles depth, intrinsics.
        - `ResizeShortestEdge` uses expanded version of `ResizeTransform`, which handles depth, intrinsics.
    """
    if not cfg.INPUT.AUG_ENABLED:
        return []
    augmentation = []
    if cfg.INPUT.CROP.ENABLED and is_train:
        augmentation.append(RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

    # Resize augmentation.
    if is_train:
        min_size = cfg.INPUT.RESIZE.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.RESIZE.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.RESIZE.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.RESIZE.MIN_SIZE_TEST
        max_size = cfg.INPUT.RESIZE.MAX_SIZE_TEST
        sample_style = "choice"
    if min_size:
        augmentation.append(ResizeShortestEdge(min_size, max_size, sample_style))

    if cfg.INPUT.RANDOM_FLIP.ENABLED and is_train:
        augmentation.append(RandomFlip())

    if cfg.INPUT.COLOR_JITTER.ENABLED and is_train:
        brightness_lower, brightness_upper = cfg.INPUT.COLOR_JITTER.BRIGHTNESS
        brightness_min, brightness_max = 1.0 - brightness_lower, 1.0 + brightness_upper
        augmentation.append(RandomBrightness(brightness_min, brightness_max))

        saturation_lower, saturation_upper = cfg.INPUT.COLOR_JITTER.SATURATION
        saturation_min, saturation_max = 1.0 - saturation_lower, 1.0 + saturation_upper
        augmentation.append(RandomSaturation(saturation_min, saturation_max))

        contrast_lower, contrast_upper = cfg.INPUT.COLOR_JITTER.CONTRAST
        contrast_min, contrast_max = 1.0 - contrast_lower, 1.0 + contrast_upper
        augmentation.append(RandomContrast(contrast_min, contrast_max))

    if not augmentation:
        LOG.warning("No Augmentation!")

    print("augmentation", augmentation)
    return augmentation
