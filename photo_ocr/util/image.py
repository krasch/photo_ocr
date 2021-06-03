from pathlib import Path
from typing import Union

import cv2
import numpy as np

from PIL import Image

from photo_ocr.util import bbdraw


def load_image(path: Union[Path, str]) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: Union[Path, str]):
    image.save(path)


def crop_and_align(image: Image.Image, box: np.array):
    """
    Crop and warp image so that it only contains the word selected by the bounding box, horizontally aligned
    :param image:
    :param box:
    :return:
    """

    box_width = int(np.linalg.norm(box[0] - box[1]) + 1)
    box_height = int(np.linalg.norm(box[1] - box[2]) + 1)

    # calculate a transformation matrix that can be used to extract only the word we are interested in right now
    destination_coordinates = np.float32([[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    transformation_matrix = cv2.getPerspectiveTransform(box, destination_coordinates)

    # perform the transformation
    # result is a cropped image containing only the word selected by the bounding box, horizontally aligned
    image = np.array(image)
    cropped = cv2.warpPerspective(image, transformation_matrix, (box_width, box_height), flags=cv2.INTER_NEAREST)
    cropped = Image.fromarray(cropped)

    return cropped


def draw_ocr_results(image, results):
    for result in results:
        polygon = [(x, y) for x, y in result.bounding_polygon]
        label = "{} ({})".format(result.word, round(result.confidence, 2))
        image = bbdraw.polygon(image, polygon, colour="green", label=label)
    return image