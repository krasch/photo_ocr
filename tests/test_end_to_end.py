import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from PIL import Image

from photo_ocr import ocr


# certainly not a unit test, just to check if photo_ocr behaves the same for different versions of required libraries
# (have been burned before by subtle, unannounced changes in deep learning libraries)
# surely could be doing something more clever here + try more than one image, but its a start
def test_end_to_end():
    expected = np.load("tests/pub.jpg.expected.npy", allow_pickle=True)

    image = Image.open("tests/pub.jpg")
    actual = ocr(image)

    assert len(actual) == len(expected)

    # sometimes there are smaller differences in the detection confidence (again I think based on opencv version)
    # since the entries are sorted by confidence, the sorting might differ between actual and expected
    # however the following step assumes that actual and expected are in the same order -> resort them here
    # sort by the (x+y) of the left-upper corner entry.polygon[0] -> most left-upper bounding box comes first
    actual = sorted(actual, key=lambda entry: sum(entry.polygon[0]))  # sum = x+y
    expected = sorted(expected, key=lambda entry: sum(entry.polygon[0]))

    for item_actual, item_expected in zip(actual, expected):
        polygon_actual, text_actual, confidence_actual = item_actual
        polygon_expected, text_expected, confidence_expected = item_expected

        # decimal=2 because quite a bit of variation based on the opencv version
        # but getting the pixel correct to 0.01 accuracy should be good enough...
        assert_array_almost_equal(polygon_actual, polygon_expected, decimal=3)
        assert_almost_equal(confidence_actual, confidence_expected, decimal=3)
        assert text_actual == text_expected

