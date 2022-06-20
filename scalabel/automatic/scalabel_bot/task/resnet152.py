import os
import torch
from pprint import pformat

from scalabel.automatic.scalabel_bot.common.logger import logger
import pipeswitch.task.common as util


MODEL_NAME = "resnet152"


class ResNet152(object):
    def __init__(self):
        self.model = None
        self.func = None
        self.shape_list = None

    def import_data(self, task_key):
        filename = "dog.jpg"
        batch_size = 8

        # Download an example image from the pytorch website
        if not os.path.isfile(filename):
            import urllib

            url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
            try:
                urllib.URLopener().retrieve(url, filename)
            except Exception:  # pylint: disable=broad-except
                urllib.request.urlretrieve(url, filename)

        # sample execution (requires torchvision)
        from PIL import Image
        from torchvision import transforms

        input_image = Image.open(filename)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        image = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        images = torch.cat([image] * batch_size)
        return images

    def import_model(self):
        # from torchvision import models

        # model = models.resnet152(pretrained=True)
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet152",
            pretrained=True,
            verbose=False,
        )
        # print(type(model))
        util.set_fullname(model, MODEL_NAME)
        logger.spam(f"\n{pformat(list(model.named_children()))}")
        # print("set_fullname")

        return model

    # TODO: figure out how model partitioning works
    def partition_model(self, model):
        group_list = []
        before_core = []
        core_complete = False
        after_core = []

        group_list.append(before_core)
        for name, child in model.named_children():
            logger.spam(f"named child: {name}, {child}")
            if "layer" in name:
                core_complete = True
                logger.spam("layer in name start")
                for name_name, child_child in child.named_children():
                    group_list.append([child_child])
                    logger.spam(f"{name_name}, {child_child}")
                logger.spam("layer in name end")
            else:
                if not core_complete:
                    before_core.append(child)
                    logger.spam(f"before core: {child}")
                else:
                    after_core.append(child)
                    logger.spam(f"after core: {child}")
        group_list.append(after_core)
        logger.spam(f"\n{pformat(group_list)}")

        return group_list
