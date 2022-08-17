# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch

from . import DefaultDatasetMapper


class NuscenesDatasetMapper(DefaultDatasetMapper):
    """
    In addition to 2D / 3D boxes, each instance also has attribute and speed.

    Assumption: image transformation does not change attributes and speed.
    """

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)

        annos = dataset_dict["annotations"]

        # NuScenes attributes
        attributes = [obj["attribute_id"] for obj in annos]
        attributes = torch.tensor(attributes, dtype=torch.int64)
        dataset_dict["instances"].gt_attributes = attributes

        # Speed (magnitude of velocity)
        speeds = [obj["speed"] for obj in annos]
        speeds = torch.tensor(speeds, dtype=torch.float32)
        dataset_dict["instances"].gt_speeds = speeds
        return dataset_dict
