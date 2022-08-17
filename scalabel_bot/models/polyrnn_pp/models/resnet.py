import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1+(dilation-1), bias=False, dilation=dilation)
        # I know it's weird to write the padding this way, just makes it clear :P
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, strides, dilations, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
            stride = strides[0], dilation = dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1],
            stride = strides[1], dilation = dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2],
            stride = strides[2], dilation = dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3],
            stride = strides[3], dilation = dilations[3])
        self.avgpool = nn.AvgPool2d(7 * max(dilations), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1,
        dilation=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_f = self.relu(x)
        x = self.maxpool(conv1_f)

        layer1_f = self.layer1(x)
        layer2_f = self.layer2(layer1_f)
        layer3_f = self.layer3(layer2_f)
        layer4_f = self.layer4(layer3_f)

        x = self.avgpool(layer4_f)
        x = x.view(x.size(0), -1)
        fc_f = self.fc(x)

        return fc_f, conv1_f, layer1_f, layer2_f, layer3_f, layer4_f