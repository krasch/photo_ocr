from PIL import Image

from torchvision import transforms

from photo_ocr.utils import transforms as custom_transforms


def calculate_resize_ratio(img: Image.Image, max_size, mag_ratio):
    # magnify image size
    target_size = mag_ratio * max(img.width, img.height)

    # but should not be larger than max size
    if target_size > max_size:
        target_size = max_size

    ratio = target_size / max(img.width, img.height)

    return ratio



def prepare_image(image, target_size, interpolation, mag_ratio=1):
    ratio = calculate_resize_ratio(image, target_size, mag_ratio)

    transform = transforms.Compose([custom_transforms.ResizeRatio(ratio),
                                    custom_transforms.PadTo32(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x2 = transform(image)
    x2 = x2.unsqueeze(0)

    return x2, 1.0 / ratio
