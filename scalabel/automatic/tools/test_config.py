from detectron2.config import get_cfg
from scalabel.automatic.model_repo import add_general_config, add_dd3d_config
import os

if __name__ == "__main__":
    cfg = get_cfg()
    add_general_config(cfg)
    add_dd3d_config(cfg)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # file_path = os.path.join(dir_path, "../model_repo/configs/dd3d/dd3d.yaml")
    file_path = os.path.join(dir_path, "../model_repo/dd3d/configs/dd3d.yaml")
    print("file_path:", file_path)
    cfg.merge_from_file(file_path)

    print("config:", cfg)
