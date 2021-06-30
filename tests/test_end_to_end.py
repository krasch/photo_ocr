import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from PIL import Image

from photo_ocr import ocr


# certainly not a unit test, just to check if photo_ocr behaves the same for different versions of required libraries
# (have been burned before by subtle, unannounced changes in deep learning libraries)
# surely could be doing something more clever here + try more than one image, but its a start
def test_end_to_end():
    expected = np.load("tests/pub.jpg.expected.npy", allow_pickle=True)

    image = Image.open("pub.jpg")
    actual = ocr(image)

    assert len(actual) == len(expected)

    for item_actual, item_expected in zip(actual, expected):
        polygon_actual, text_actual, confidence_actual = item_actual
        polygon_expected, text_expected, confidence_expected = item_expected

        assert_array_almost_equal(polygon_actual, polygon_expected, decimal=3)
        assert_almost_equal(confidence_actual, confidence_expected, decimal=3)
        assert text_actual == text_expected

