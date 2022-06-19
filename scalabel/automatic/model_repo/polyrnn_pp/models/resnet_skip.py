import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms

from .resnet import ResNet, Bottleneck


class SkipResnet50(nn.Module):
    def __init__(self, concat_channels=64, final_dim=128):
        super(SkipResnet50, self).__init__()

        # Default transform for all torchvision models
        self.normalizer = transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.concat_channels = concat_channels
        self.final_dim = final_dim
        self.feat_size = 28

        self.image_feature_dim = 128
        self.resnet = ResNet(Bottleneck, layers=[3, 4, 6, 3], strides=[1, 2, 1, 1],
            dilations=[1, 1, 2, 4])

        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)

        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)

        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.res1_concat = nn.Sequential(concat2, bn2, relu2, up2)

        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        self.res2_concat = nn.Sequential(concat3, bn3, relu3, up3)

        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        self.res4_concat = nn.Sequential(concat4, bn4, relu4, up4)

        # Different from original, original used maxpool
        # Original used no activation here
        conv_final_1 = nn.Conv2d(4*concat_channels, 128, kernel_size=3, padding=1, stride=2,
            bias=False)
        bn_final_1 = nn.BatchNorm2d(128)
        conv_final_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False)
        bn_final_2 = nn.BatchNorm2d(128)
        conv_final_3 = nn.Conv2d(128, final_dim, kernel_size=3, padding=1, bias=False)
        bn_final_3 = nn.BatchNorm2d(final_dim)

        self.conv_final = nn.Sequential(conv_final_1, bn_final_1, conv_final_2, bn_final_2,
            conv_final_3, bn_final_3)

    def forward(self, x):
        x = self.normalize(x)
        # Normalization

        fc_f, conv1_f, layer1_f, layer2_f, layer3_f, layer4_f = self.resnet(x)

        conv1_f = self.conv1_concat(conv1_f)
        layer1_f = self.res1_concat(layer1_f)
        layer2_f = self.res2_concat(layer2_f)
        layer4_f = self.res4_concat(layer4_f)

        concat_features = torch.cat((conv1_f, layer1_f, layer2_f, layer4_f), dim=1)

        final_features = self.conv_final(concat_features)

        return concat_features, final_features

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)


if __name__ == '__main__':
    model = SkipResnet50()
    print([a.size() for a in model(torch.randn(1,3,224,224))])
