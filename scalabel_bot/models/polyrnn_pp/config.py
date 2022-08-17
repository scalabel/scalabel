from detectron2.config import CfgNode as CN


def add_polyrnnpp_config(cfg):
    """
    Add config for Polygon-RNN++.
    """
    cfg.MODEL.META_ARCHITECTURE = "PolyRNNPP"

    cfg.MODEL.POLYRNNPP = CN()
    cfg.MODEL.POLYRNNPP.MAX_POLY_LEN = 71
    cfg.MODEL.POLYRNNPP.USE_BN_LSTM = True
