from detectron2.config import CfgNode as CN


def add_general_config(cfg):
    """
    Add config for Polygon-RNN++.
    """
    cfg.TASK_TYPE = "OD"
