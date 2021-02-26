from typing import Union
from pathlib import Path

import cv2
from PIL import Image

from photo_ocr.detection.detect import TextDetector
from photo_ocr.utils import bbdraw

# load all the necessary models
detector = TextDetector()


# utility method for convenience
def load_image(path: Union[Path, str]) -> Image.Image:
    return Image.open(path).convert("RGB")


# utility method for convenience / consistency
def save_image(image: Image.Image, path: Union[Path, str]):
    image.save(path)


# todo comments explaining the parameters
def perform_ocr(image: Image.Image,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4,
                refine=True,
                canvas_size=1280,
                mag_ratio=1.5,
                interpolation=cv2.INTER_LINEAR):

    boxes = detector.detect(image,
                            text_threshold, link_threshold, low_text, refine,
                            canvas_size, mag_ratio, interpolation)
    return boxes


def draw_polygons(image, polys):
    for i, poly in enumerate(polys):
        poly = [(x, y) for x, y in poly]
        image = bbdraw.polygon(image, poly, colour="green")
    return image





