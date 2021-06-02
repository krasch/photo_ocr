"""
Methods to "find" in the raw detection outputs the pixels that form individual words (represented as segmentation maps)
"""
from collections import namedtuple
from typing import List, Tuple

import cv2
import numpy as np


Component = namedtuple("Component", ["centroid_x", "centroid_y",
                                     "left", "top", "width", "height",
                                     "size", "segmentation"])


def _parse_connected_components(components):
    """
    Makes output from cv2.connectedComponentsWithStats more accessible
    :param components: output from cv2.connectedComponentsWithStats (tuple of num_components, labels, stats, centroids)
    :return: iterator over parsed components
    """
    num_components, labels, stats, centroids = components

    # 0 = background -> ignore, start loop at i=1
    for i in range(1, num_components):
        component = Component(centroid_x=centroids[i][0],
                              centroid_y=centroids[i][1],
                              left=stats[i, cv2.CC_STAT_LEFT],
                              top=stats[i, cv2.CC_STAT_TOP],
                              width=stats[i, cv2.CC_STAT_WIDTH],
                              height=stats[i, cv2.CC_STAT_HEIGHT],
                              size=stats[i, cv2.CC_STAT_AREA],
                              segmentation=(labels == i))
        yield component


def get_word_segmentations(text_scores: np.array,
                           link_scores: np.array,
                           text_threshold_first_pass: float,
                           text_threshold_second_pass: float,
                           link_threshold: float) -> Tuple[List[Component], np.array]:
    """
    Based on the text_scores and link_scores, calculate connected components -- each of which representing a
    word in the original image. Returns for each component/word, a segmentation map with True = pixel is either part
    of a character of this word or is part of the link area between the characters of this word. Also returns a
    segmentation map that marks with True all the pixels that are part of the link area of any word.
    :param text_scores: For each pixel, how likely is it that this pixel is part of a text character
    :param link_scores: For each pixel, how likely it is that this pixel is between two text characters
    :param text_threshold_first_pass:
    :param text_threshold_second_pass:
    :param link_threshold:
    :return:
    """

    # only use pixels that are over the respective thresholds
    _, text_area = cv2.threshold(text_scores.copy(), text_threshold_first_pass, 1, 0)
    _, link_area = cv2.threshold(link_scores.copy(), link_threshold, 1, 0)

    # this is the area between characters
    link_area = np.logical_and(link_area == 1, text_area == 0)

    # these are areas that are either characters or links between characters
    # -> a word is a bunch of characters that are linked together
    text_or_link_area = np.clip(text_area + link_area, 0, 1)

    # to find the words: need to find connected components of (characters, links)
    components = cv2.connectedComponentsWithStats(text_or_link_area.astype(np.uint8), connectivity=4)
    components = _parse_connected_components(components)

    # remove components that contain fewer than 10 pixels
    components = [component for component in components if component.size >= 10]

    # final round of thresholding
    def component_is_below_threshold(component):
        return np.max(text_scores[component.segmentation]) < text_threshold_second_pass
    components = [component for component in components if not component_is_below_threshold(component)]

    return components, link_area

