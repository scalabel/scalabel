# Copyright 2021 Toyota Research Institute.  All rights reserved.
# pylint: disable=no-value-for-parameter, redundant-keyword-arg
from .dataset_mapper import DefaultDatasetMapper
from .nuscenes_mapper import NuscenesDatasetMapper


def get_dataset_mapper(cfg, is_train=True):
    if is_train:
        dataset_mapper_name = cfg.DATASETS.TRAIN.DATASET_MAPPER
    else:
        dataset_mapper_name = cfg.DATASETS.TEST.DATASET_MAPPER

    if dataset_mapper_name == "default":
        return DefaultDatasetMapper(cfg, is_train=is_train)
    elif dataset_mapper_name == "nuscenes":
        return NuscenesDatasetMapper(cfg, is_train=is_train)
    else:
        raise ValueError(f"Invalid dataset mapper: {dataset_mapper_name}")
