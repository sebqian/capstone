"""Taken and modified from https://github.com/ChoiDM/pytorch-deeplabv3plus-3D """

import torch.nn as nn
import torch.nn.functional as F

from codebase.projects.hecktor2022.models.deeplabv3 import resnet_deeplab
from codebase.projects.hecktor2022.models import aspp


class DeepLabV3_3D(nn.Module):
    def __init__(self, num_classes, input_channels, resnet_name, last_activation=None):
        super(DeepLabV3_3D, self).__init__()
        self.num_classes = num_classes
        self.last_activation = last_activation

        if resnet_name.lower() == 'resnet18_os16':
            self.resnet = resnet_deeplab.ResNet18_OS16(input_channels)

        elif resnet_name.lower() == 'resnet34_os16':
            self.resnet = resnet_deeplab.ResNet34_OS16(input_channels)

        elif resnet_name.lower() == 'resnet50_os16':
            self.resnet = resnet_deeplab.ResNet50_OS16(input_channels)

        elif resnet_name.lower() == 'resnet101_os16':
            self.resnet = resnet_deeplab.ResNet101_OS16(input_channels)

        elif resnet_name.lower() == 'resnet152_os16':
            self.resnet = resnet_deeplab.ResNet152_OS16(input_channels)

        elif resnet_name.lower() == 'resnet18_os8':
            self.resnet = resnet_deeplab.ResNet18_OS8(input_channels)

        elif resnet_name.lower() == 'resnet34_os8':
            self.resnet = resnet_deeplab.ResNet34_OS8(input_channels)

        if resnet_name.lower() in ['resnet50_os16', 'resnet101_os16', 'resnet152_os16']:
            self.aspp = aspp.ASPP_Bottleneck(num_classes=self.num_classes)
        else:
            self.aspp = aspp.ASPP(num_classes=self.num_classes)

    def forward(self, x):

        h = x.size()[2]
        w = x.size()[3]
        c = x.size()[4]

        feature_map = self.resnet(x)

        output = self.aspp(feature_map)

        output = F.interpolate(output, size=(h, w, c), mode='trilinear', align_corners=True)

        if self.last_activation.lower() == 'sigmoid':
            output = nn.Sigmoid()(output)

        elif self.last_activation.lower() == 'softmax':
            output = nn.Softmax()(output)

        return output
