from PIL import Image
from torchvision.transforms import functional as F
import cv2
import numpy as np


class PadTo32:
    def __init__(self, fill=0, padding_mode="constant"):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img: Image.Image):

        pad_right = 32 - img.width % 32
        if pad_right == 32:
            pad_right = 0

        pad_bottom = 32 - img.height % 32
        if pad_bottom == 32:
            pad_bottom = 0

        return F.pad(img, (0, 0, pad_right, pad_bottom), self.fill, self.padding_mode)


class ResizeRatio:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img: Image.Image):
        # todo use PIL
        #target_size = int(img.height * self.ratio), int(img.width * self.ratio)
        #return F.resize(img, target_size, interpolation=Image.BILINEAR)

        target_h, target_w = int(img.height * self.ratio), int(img.width * self.ratio)
        resized = cv2.resize(np.array(img), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(resized)



