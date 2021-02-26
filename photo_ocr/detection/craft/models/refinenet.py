"""
Copyright (c) 2019-present NAVER Corp.
"""


import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

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
    "refine_net": "https://photoocr.s3-eu-west-1.amazonaws.com/craft_refiner_CTW1500.pth",
}


# This class is a copy from refinenet.py in the original repository, no changes.
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.last_conv = nn.Sequential(
            nn.Conv2d(34, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    def forward(self, y, upconv4):
        refine = torch.cat([y.permute(0,3,1,2), upconv4], dim=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        #out = torch.add([aspp1, aspp2, aspp3, aspp4], dim=1)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)  # , refine.permute(0,2,3,1)


# new code to match other models in torchvision model zoo
def refine_net(pretrained, progress, **kwargs):
    model = RefineNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['refine_net'], progress=progress)
        model.load_state_dict(state_dict)

    return model


