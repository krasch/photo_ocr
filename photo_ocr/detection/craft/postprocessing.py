"""
Methods to postprocess CRAFT model outputs.
The code corresponds to code from the original repository (./craft_utils.py),
but has been extensively refactored to improve clarity.
"""
import numpy as np
import cv2
import math

from photo_ocr.utils.cv2_utils import parse_connected_components, Component
from photo_ocr.detection.craft.postprocessing_polygons import calculate_polygon


def _calculate_word_segmentation_map(component: Component, link_area_map: np.array) -> np.array:
    """
    Create a segmentation map such that: letters of the detected word = white, background = black
    :param component: information about the connected component representing the word
    :param link_area_map: information which pixels are areas _between_ letters
    :return: segmentation map with size of the original image
    """

    # step 1: initialize segmentation with raw pixels of the connected component
    segmentation_map = np.zeros(component.segmentation.shape, dtype=np.uint8)
    segmentation_map[component.segmentation] = 255

    # step 2: raw pixels also include area between the letters (link area), mark this area as background (black)
    segmentation_map[link_area_map] = 0

    # step 3: the letters have irregular shapes, turn them into something more approaching rectangular shape
    # calculation of kernel size taken from original code, there was no explanation why it was calculated this way
    ratio = component.size * min(component.width, component.height) / (component.width * component.height)
    kernel_size = int(math.sqrt(ratio) * 2)

    # calculate dilation area
    left = component.left - kernel_size
    right = component.left + component.width + kernel_size + 1
    top = component.top - kernel_size
    bottom = component.top + component.height + kernel_size + 1

    # dilation area should not be outside image boundaries
    image_width, image_height = segmentation_map.shape
    left = max(0, left)
    right = min(image_height, right)
    top = max(0, top)
    bottom = min(image_width, bottom)

    # perform the dilation (i.e. turn the letters more rectangular)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + kernel_size, 1 + kernel_size))  # 1+ sic!...
    segmentation_map[top:bottom, left:right] = cv2.dilate(segmentation_map[top:bottom, left:right], kernel)

    return segmentation_map


def _calculate_bounding_box(segmentation: np.array) -> np.array:
    """
    Calculates a bounding box around a connected component (representation by a segmentation map)
    :param segmentation: array the size of original image, black (255) = component, white (0) = background
    :return: bounding box of [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], starting with corner with minimum x+y
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
    # (comment from original code: "align diamond-shape")
    box_width, box_height = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    box_ratio = max(box_width, box_height) / (min(box_width, box_height) + 1e-5)
    if abs(1 - box_ratio) <= 0.1:
        left, right = min(contour[:, 0]), max(contour[:, 0])
        top, bottom = min(contour[:, 1]), max(contour[:, 1])
        box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)

    # corners of box should be clockwise, starting with corner with minimum x+y
    start_index = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - start_index, 0)
    box = np.array(box)

    return box


def calculate_bounding_boxes_and_polygons(textmap, linkmap, text_threshold, link_threshold, low_text):

    # apply thresholds
    _, text_score = cv2.threshold(textmap.copy(), low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap.copy(), link_threshold, 1, 0)

    # this is the area between letters, will want to remove it later so that we keep only the letters
    link_area = np.logical_and(link_score == 1, text_score == 0)

    # these are areas that are either letter or links between letters
    # -> a word is a bunch of letters that are linked together
    letter_or_link_area = np.clip(text_score + link_score, 0, 1)

    # to find the words: need to find connected components of (letters, links)
    components = cv2.connectedComponentsWithStats(letter_or_link_area.astype(np.uint8), connectivity=4)
    components = parse_connected_components(components)

    for component in components:
        # size filtering
        if component.size < 10:
            continue

        # thresholding
        if np.max(textmap[component.segmentation]) < text_threshold:
            continue

        # create a segmentation map such that: letters of the detected word = white, background = black
        # needs the link_area to remove the pixels that are links between letters
        segmentation_map = _calculate_word_segmentation_map(component, link_area)

        # calculate a bounding box around the connected component based on segmentation map
        # clockwise four corners of [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], starting with corner with minimum x+y
        box = _calculate_bounding_box(segmentation_map)

        polygon = calculate_polygon(box, component.segmentation)

        yield box, polygon


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    boxes, polys = zip(*calculate_bounding_boxes_and_polygons(textmap, linkmap, text_threshold, link_threshold, low_text))
    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
