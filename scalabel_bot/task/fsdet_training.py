import argparse
import os
import time

import scalabel_bot.task.common as util
from scalabel_bot.task.fsdet import FSDET
import torch
import torch.nn as nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from scalabel_bot.models.few_shot_detection.fsdet.config import (
    get_cfg,
    set_global_cfg,
)
from scalabel_bot.models.few_shot_detection.fsdet.engine import (
    default_argument_parser,
    default_setup,
)
from scalabel_bot.models.few_shot_detection.fsdet.evaluation import (
    verify_results,
)
from scalabel_bot.models.few_shot_detection.tools.ckpt_surgery import (
    ckpt_surgery,
    combine_ckpts,
)
from scalabel_bot.models.few_shot_detection.tools.train_net import (
    Trainer,
)


TASK_NAME = "fsdet_training"


class FSDETTraining:
    def __init__(self):
        self.fsdet = FSDET()

    def setup(self, args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        if args.opts:
            cfg.merge_from_list(args.opts)
        cfg.freeze()
        set_global_cfg(cfg)
        default_setup(cfg, args)
        return cfg

    def get_parser(self):
        parser = argparse.ArgumentParser()
        # Paths
        parser.add_argument(
            "--src1", type=str, default="", help="Path to the main checkpoint"
        )
        parser.add_argument(
            "--src2",
            type=str,
            default="",
            help="Path to the secondary checkpoint (for combining)",
        )
        parser.add_argument(
            "--save-dir", type=str, default="", help="Save directory"
        )
        # Surgery method
        parser.add_argument(
            "--method",
            choices=["combine", "remove", "randinit"],
            required=True,
            help=(
                "Surgery method. combine = "
                "combine checkpoints. remove = for fine-tuning on "
                "novel dataset, remove the final layer of the "
                "base detector. randinit = randomly initialize "
                "novel weights."
            ),
        )
        # Targets
        parser.add_argument(
            "--param-name",
            type=str,
            nargs="+",
            default=[
                "roi_heads.box_predictor.cls_score",
                "roi_heads.box_predictor.bbox_pred",
            ],
            help="Target parameter names",
        )
        parser.add_argument(
            "--tar-name",
            type=str,
            default="model_reset",
            help="Name of the new ckpt",
        )
        # Dataset
        parser.add_argument(
            "--coco", action="store_true", help="For COCO models"
        )
        parser.add_argument(
            "--lvis", action="store_true", help="For LVIS models"
        )

        return parser

    def import_data(self, task):
        return self.fsdet.import_data(task)

    def import_model(self):
        model = self.fsdet.import_model()
        model.train()
        return model

    def import_func(self):
        def train(model, data_loader):
            # ckpt surgery
            args = get_parser().parse_args(
                [
                    "--src1",
                    "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth",
                    "--method",
                    "randinit",
                    "--save-dir",
                    "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all--coco",
                ]
            )

            # COCO
            if args.coco:
                # COCO
                # fmt: off
                args.NOVEL_CLASSES = [
                    1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67,
                    72,
                ]
                args.BASE_CLASSES = [
                    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
                    36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                    55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
                    81, 82, 84, 85, 86, 87, 88, 89, 90,
                ]
                # fmt: on
                args.ALL_CLASSES = sorted(
                    args.BASE_CLASSES + args.NOVEL_CLASSES
                )
                args.IDMAP = {v: i for i, v in enumerate(args.ALL_CLASSES)}
                args.TAR_SIZE = 80
            elif args.lvis:
                # LVIS
                # fmt: off
                args.NOVEL_CLASSES = [
                    0, 6, 9, 13, 14, 15, 20, 21, 30, 37, 38, 39, 41, 45, 48, 50, 51, 63,
                    64, 69, 71, 73, 82, 85, 93, 99, 100, 104, 105, 106, 112, 115, 116,
                    119, 121, 124, 126, 129, 130, 135, 139, 141, 142, 143, 146, 149,
                    154, 158, 160, 162, 163, 166, 168, 172, 180, 181, 183, 195, 198,
                    202, 204, 205, 208, 212, 213, 216, 217, 218, 225, 226, 230, 235,
                    237, 238, 240, 241, 242, 244, 245, 248, 249, 250, 251, 252, 254,
                    257, 258, 264, 265, 269, 270, 272, 279, 283, 286, 290, 292, 294,
                    295, 297, 299, 302, 303, 305, 306, 309, 310, 312, 315, 316, 317,
                    319, 320, 321, 323, 325, 327, 328, 329, 334, 335, 341, 343, 349,
                    350, 353, 355, 356, 357, 358, 359, 360, 365, 367, 368, 369, 371,
                    377, 378, 384, 385, 387, 388, 392, 393, 401, 402, 403, 405, 407,
                    410, 412, 413, 416, 419, 420, 422, 426, 429, 432, 433, 434, 437,
                    438, 440, 441, 445, 453, 454, 455, 461, 463, 468, 472, 475, 476,
                    477, 482, 484, 485, 487, 488, 492, 494, 495, 497, 508, 509, 511,
                    513, 514, 515, 517, 520, 523, 524, 525, 526, 529, 533, 540, 541,
                    542, 544, 547, 550, 551, 552, 554, 555, 561, 563, 568, 571, 572,
                    580, 581, 583, 584, 585, 586, 589, 591, 592, 593, 595, 596, 599,
                    601, 604, 608, 609, 611, 612, 615, 616, 625, 626, 628, 629, 630,
                    633, 635, 642, 644, 645, 649, 655, 657, 658, 662, 663, 664, 670,
                    673, 675, 676, 682, 683, 685, 689, 695, 697, 699, 702, 711, 712,
                    715, 721, 722, 723, 724, 726, 729, 731, 733, 734, 738, 740, 741,
                    744, 748, 754, 758, 764, 766, 767, 768, 771, 772, 774, 776, 777,
                    781, 782, 784, 789, 790, 794, 795, 796, 798, 799, 803, 805, 806,
                    807, 808, 815, 817, 820, 821, 822, 824, 825, 827, 832, 833, 835,
                    836, 840, 842, 844, 846, 856, 862, 863, 864, 865, 866, 868, 869,
                    870, 871, 872, 875, 877, 882, 886, 892, 893, 897, 898, 900, 901,
                    904, 905, 907, 915, 918, 919, 920, 921, 922, 926, 927, 930, 931,
                    933, 939, 940, 944, 945, 946, 948, 950, 951, 953, 954, 955, 956,
                    958, 959, 961, 962, 963, 969, 974, 975, 988, 990, 991, 998, 999,
                    1001, 1003, 1005, 1008, 1009, 1010, 1012, 1015, 1020, 1022, 1025,
                    1026, 1028, 1029, 1032, 1033, 1046, 1047, 1048, 1049, 1050, 1055,
                    1066, 1067, 1068, 1072, 1073, 1076, 1077, 1086, 1094, 1099, 1103,
                    1111, 1132, 1135, 1137, 1138, 1139, 1140, 1144, 1146, 1148, 1150,
                    1152, 1153, 1156, 1158, 1165, 1166, 1167, 1168, 1169, 1171, 1178,
                    1179, 1180, 1186, 1187, 1188, 1189, 1203, 1204, 1205, 1213, 1215,
                    1218, 1224, 1225, 1227
                ]
                # fmt: on
                args.BASE_CLASSES = [
                    c for c in range(1230) if c not in args.NOVEL_CLASSES
                ]
                args.ALL_CLASSES = sorted(
                    args.BASE_CLASSES + args.NOVEL_CLASSES
                )
                args.IDMAP = {v: i for i, v in enumerate(args.ALL_CLASSES)}
                args.TAR_SIZE = 1230
            else:
                # VOC
                args.TAR_SIZE = 20

            if args.method == "combine":
                combine_ckpts(args)
            else:
                ckpt_surgery(args)

            # train_net
            args = default_argument_parser().parse_args(
                [
                    "--num-gpus",
                    "1",
                    "--config-file",
                    "configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml",
                    "--opts",
                    "MODEL_WEIGHTS",
                    "few_shot_detection/checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1/model_reset_surgery.pth",
                ]
            )

            cfg = self.setup(args)

            if args.eval_only:
                model = Trainer.build_model(cfg)
                DetectionCheckpointer(
                    model, save_dir=cfg.OUTPUT_DIR
                ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
                res = Trainer.test(cfg, model)
                if comm.is_main_process():
                    verify_results(cfg, res)
                return res

            """
            If you'd like to do anything fancier than the standard training logic,
            consider writing your own training loop or subclassing the trainer.
            """
            trainer = Trainer(cfg)
            trainer.resume_or_load(resume=args.resume)
            return trainer.train()

        return train

    def import_task(self):
        model = self.import_model()
        func = self.import_func()
        return func
