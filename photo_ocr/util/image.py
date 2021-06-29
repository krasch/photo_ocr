from pathlib import Path
from typing import Union

from PIL import Image, ImageOps
import cv2
import numpy as np
import bbdraw


def load_image(path: Union[Path, str]) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image


def _sort_clockwise(box: np.array) -> np.array:
    """
    Re-order corners of box to be clockwise, starting with corner with minimum x+y
    :param box:
    :return:
    """
    start_index = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - start_index, 0)
    box = np.array(box)
    return box


def crop_and_align(image: Image.Image, polygon: np.array):
    """
    Crop and warp image so that it only contains the word selected by the bounding polygon, horizontally aligned
    :param image:
    :param polygon:
    :return:
    """

    polygon = np.array(polygon).astype(np.float32)
    rect = cv2.minAreaRect(polygon)

    box = cv2.boxPoints(rect)
    box = _sort_clockwise(box)

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
    # draw most-confident results last, so they are on top
    results = sorted(results, key=lambda item: item.confidence)

    for result in results:
        label = "{} ({:.2f})".format(result.text, result.confidence)
        image = bbdraw.polygon(image, result.polygon, color=bbdraw.bbdraw.PURPLE, text=label)
    return image
