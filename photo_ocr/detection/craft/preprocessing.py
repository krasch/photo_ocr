from PIL import Image

from torchvision import transforms
from torchvision.transforms import functional as F


class ResizeRatio:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img: Image.Image):
        target_size = int(img.height * self.ratio), int(img.width * self.ratio)
        return F.resize(img, target_size, interpolation=Image.BILINEAR)


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


def calculate_resize_ratio(img: Image.Image, max_size, mag_ratio):
    # magnify image size
    target_size = mag_ratio * max(img.width, img.height)

    # but should not be larger than max size
    if target_size > max_size:
        target_size = max_size

    ratio = target_size / max(img.width, img.height)

    return ratio


def init_transforms(resize_ratio):
    return transforms.Compose([ResizeRatio(resize_ratio),
                               PadTo32(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
