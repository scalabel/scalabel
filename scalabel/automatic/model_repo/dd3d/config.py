from detectron2.config import CfgNode as CN


def add_dd3d_config(cfg):
    """
    Add config for DD3D.
    """
    # print("cfg", cfg)
    cfg.DATASETS.TRAIN = CN()
    cfg.DATASETS.TEST = CN()
    cfg.INPUT.RANDOM_FLIP = CN()
    cfg.set_new_allowed(True)
