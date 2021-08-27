import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from PIL import Image

from photo_ocr import ocr


def assert_ocr_results_almost_equal(actual, expected):
    # utility function for sorting
    def left_upper_corner_x_plus_y(ocr_result):
        polygon, text, confidence = ocr_result
        x, y = polygon[0]  # polygons are always arranged so that left-upper corner comes first
        return x + y

    assert len(actual) == len(expected)

    # sometimes there are smaller differences in the detection confidence (again I think based on opencv version)
    # since the entries are sorted by confidence, the sorting might differ between actual and expected
    # however the following step assumes that actual and expected are in the same order
    # -> resort them here by left-upper corner of polygon (smallest comes first)
    actual = sorted(actual, key=left_upper_corner_x_plus_y)
    expected = sorted(expected, key=left_upper_corner_x_plus_y)

    for item_actual, item_expected in zip(actual, expected):
        polygon_actual, text_actual, confidence_actual = item_actual
        polygon_expected, text_expected, confidence_expected = item_expected

        # decimal=2 because quite a bit of variation based on the opencv version
        # but getting the pixel correct to 0.01 accuracy should be good enough...
        assert_array_almost_equal(polygon_actual, polygon_expected, decimal=2)
        assert_almost_equal(confidence_actual, confidence_expected, decimal=2)
        assert text_actual == text_expected


# certainly not a unit test, just to check if photo_ocr behaves the same for different versions of required libraries
# (have been burned before by subtle, unannounced changes in deep learning libraries)
# surely could be doing something more clever here + try more than one image, but its a start
def test_one_image():
    expected = np.load("tests/pub.jpg.expected.npy", allow_pickle=True)

    image = Image.open("tests/pub.jpg")
    actual = ocr(image)

    assert_ocr_results_almost_equal(actual, expected)


# test to make sure batch processing of images works
def test_multiple_images():
    expected1 = np.load("tests/stickers.jpg.expected.npy", allow_pickle=True)
    expected2 = np.load("tests/pub.jpg.expected.npy", allow_pickle=True)

    image1 = Image.open("tests/stickers.jpg")
    image2 = Image.open("tests/pub.jpg")

    actual1, actual2 = ocr([image1, image2])

    assert_ocr_results_almost_equal(actual1, expected1)
    assert_ocr_results_almost_equal(actual2, expected2)

