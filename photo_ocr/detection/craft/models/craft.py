"""
Copyright (c) 2019-present NAVER Corp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

from photo_ocr.detection.craft.models.basenet import VGG16BN
from photo_ocr.detection.craft.models.utils import init_weights

# 1. These weights use different layer names than the original ones. Why?
# The original repository contains links to weights in a google drive.
# It seems like the original weights were saved with different layer names than the final code, so loading
# the weights into a model, it was first necessary to rename some layers.
# To avoid all this extra code, stored the weights with the correct names.
# 2. How the weights were stored
# _use_new_zipfile_serialization=True because load_state_dict_from_url broken with new zipfile format
# 3. TL;DR:
# -> these weights are exactly the same as the original ones, just use the correct layer names
model_urls = {
    "craft": "https://photoocr.s3-eu-west-1.amazonaws.com/craft_mlt_25k.pth",
}


# This class is a copy from craft.py in the original repository (renamed from snake_case to CamelCase).
class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# This class is a copy from craft.py in the original repository, no changes.
class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = VGG16BN(pretrained, freeze)

        """ U network """
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


# new code to match other models in torchvision model zoo
def craft(pretrained, progress, **kwargs):
    model = CRAFT(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['craft'], progress=progress)
        model.load_state_dict(state_dict)

    return model
