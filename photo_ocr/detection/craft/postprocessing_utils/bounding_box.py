"""
Methods for calculating bounding boxes around the connected components found during the segmentation step.
"""

import numpy as np
import cv2
import math

from photo_ocr.detection.craft.postprocessing_utils.segmentation import Component


def _dilate_characters(segmentation: np.array, component: Component) -> np.array:
    """
    The characters have irregular shapes, turn them into something more approaching rectangular shape
    :param segmentation:
    :param component:
    :return:
    """
    # calculation of kernel size taken from original code, there was no explanation why it was calculated this way
    ratio = component.size * min(component.width, component.height) / (component.width * component.height)
    kernel_size = int(math.sqrt(ratio) * 2)

    # calculate dilation area
    left = component.left - kernel_size
    right = component.left + component.width + kernel_size + 1
    top = component.top - kernel_size
    bottom = component.top + component.height + kernel_size + 1

    # dilation area should not be outside image boundaries
    image_width, image_height = segmentation.shape
    left = max(0, left)
    right = min(image_height, right)
    top = max(0, top)
    bottom = min(image_width, bottom)

    # perform the dilation (i.e. turn the letters more rectangular)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + kernel_size, 1 + kernel_size))  # 1+ sic!...
    segmentation[top:bottom, left:right] = cv2.dilate(segmentation[top:bottom, left:right], kernel)

    return segmentation


def _calculate_box(segmentation: np.array) -> np.array:
    """
    Calculates a bounding box around the word represented by the segmentation map
    :param segmentation:
    :return:
    """
    # calculate contour vertices
    contour = np.array(np.where(segmentation != 0))     # -> [[y1, y2, y3, ...], [x1, x2, x3, ...]]
    contour = np.roll(contour, 1, axis=0)               # -> [[x1, x2, x3, ...], [y1, y2, y3, ...]]
    contour = contour.transpose().reshape(-1, 2)        # -> [[x1, y1], [x2, y2], [x3, y3], ...]]

    # calculate box around contour
    rectangle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rectangle)

    # this does something with boxes that are roughly square
    # why this is done is unclear
    # (comment from original code: "align diamond-shape") todo
    box_width, box_height = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    box_ratio = max(box_width, box_height) / (min(box_width, box_height) + 1e-5)
    if abs(1 - box_ratio) <= 0.1:
        left, right = min(contour[:, 0]), max(contour[:, 0])
        top, bottom = min(contour[:, 1]), max(contour[:, 1])
        box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)

    return box


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


def calculate_bounding_box(component: Component, link_area: np.array):
    """
    Calculate the bounding box around the word represented by the component.
    :param component:
    :param link_area:
    :return:
    """
    # detected word = 1, background = 0
    segmentation = component.segmentation.astype(np.uint8)

    # only want the actual characters, mark space between the letters as background
    segmentation[link_area] = 0

    #  make characters more rectangular (why??)
    segmentation = _dilate_characters(segmentation, component)

    # calculate a bounding box around the connected component based on segmentation map
    box = _calculate_box(segmentation)

    # clockwise four corners of [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], starting with corner with minimum x+y
    box = _sort_clockwise(box)

    return box
