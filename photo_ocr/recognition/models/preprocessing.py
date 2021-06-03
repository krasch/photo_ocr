import math

from PIL import Image
from torchvision import transforms as transforms


class ResizeRatio:
    def __init__(self, max_size):
        self.target_height, self.target_width = max_size

    def __call__(self, image: Image.Image):
        current_width, current_height = image.size
        current_ratio = current_width / float(current_height)

        # how wide would the image be if resized to match target height?
        new_width = math.ceil(self.target_height * current_ratio)

        # image would be too wide -> cap the width (=messes up the ratio)
        if new_width > self.target_width:
            new_width = self.target_width

        resized_image = image.resize((new_width, self.target_height), Image.BICUBIC)
        return resized_image


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

    if keep_ratio:
        return transforms.Compose([transforms.Grayscale(),
                                   ResizeRatio((target_height, target_width)),
                                   PadRight(target_width),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=0.5, std=0.5)])
    else:
        return transforms.Compose([transforms.Grayscale(),
                                   transforms.Resize((target_height, target_width), interpolation=Image.BICUBIC),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=0.5, std=0.5)])

