from typing import Callable, Generator, Tuple

import numpy as np

from photo_ocr.detection.craft.postprocessing_utils.segmentation import get_word_segmentations
from photo_ocr.detection.craft.postprocessing_utils.bounding_box import calculate_bounding_box
from photo_ocr.detection.craft.postprocessing_utils.polygon import calculate_polygon, PolygonCalculationError


def init_postprocessing(text_threshold_first_pass: float,
                        text_threshold_second_pass: float,
                        link_threshold: float) -> Callable:

    def postprocess(score_text: np.array, score_link: np.array) -> Generator[np.array, None, None]:
        """
        Find the bounding boxes and polygons around every detected word.
        :param score_text: for each pixel, how likely is it that this pixel is part of a text character
        :param score_link: for each pixel, how likely it is that this pixel is between two text characters
        :return:
        """

        words, link_area = get_word_segmentations(score_text, score_link,
                                                  text_threshold_first_pass, text_threshold_second_pass, link_threshold)

        for word in words:

            # calculate an (angled) bounding box around the word
            bounding_box = calculate_bounding_box(word, link_area)

            # calculate a tight polygon around the word
            try:
                polygon = calculate_polygon(word.segmentation, bounding_box)
            except PolygonCalculationError:
                # polygon calculation is pretty restrictive, e.g. does not work on very small bounding boxes
                # if it was not possible to find a polygon, just fall back to the bounding box
                # (which is also a polygon after all..)
                polygon = bounding_box

            yield polygon

    return postprocess
