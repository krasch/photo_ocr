import math

from PIL import Image
from torchvision import transforms as transforms
from torchvision.transforms import functional as F


class ResizeRatio:
    def __init__(self, max_size, interpolation):
        self.target_height, self.target_width = max_size
        self.interpolation = interpolation

    def __call__(self, image: Image.Image):
        current_width, current_height = image.size
        current_ratio = current_width / float(current_height)

        # how wide would the image be if resized to match target height?
        new_width = math.ceil(self.target_height * current_ratio)

        # image would be too wide -> cap the width (=messes up the ratio)
        if new_width > self.target_width:
            new_width = self.target_width

        return F.resize(image, (self.target_height, new_width), interpolation=self.interpolation)


class PadRight(object):

    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, image: Image.Image):
        current_width, _ = image.size

        # nothing to do
        if current_width == self.target_width:
            return image

        # image is already wider than target with, can't pad
        if current_width > self.target_width:
            raise ValueError("Image is already wider than target with, can't pad")

        pad = self.target_width - current_width
        transform = transforms.Pad((0, 0, pad, 0), padding_mode="edge")

        return transform(image)


def init_transforms(image_shape, keep_ratio):
    target_height, target_width, _ = image_shape

    try:
        # torchvision > 0.8 uses custom interpolation modes
        interpolation = transforms.InterpolationMode.BICUBIC
    except AttributeError:
        # torchvision <= 0.8 used PIL interpolation modes
        interpolation = Image.BICUBIC

    if keep_ratio:
        return transforms.Compose([transforms.Grayscale(),
                                   ResizeRatio((target_height, target_width), interpolation=interpolation),
                                   PadRight(target_width),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=0.5, std=0.5)])
    else:
        return transforms.Compose([transforms.Grayscale(),
                                   transforms.Resize((target_height, target_width), interpolation=interpolation),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=0.5, std=0.5)])

