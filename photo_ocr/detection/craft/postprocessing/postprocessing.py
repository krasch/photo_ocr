from photo_ocr.detection.craft.postprocessing.segmentation import get_word_segmentations
from photo_ocr.detection.craft.postprocessing.bounding_box import calculate_bounding_box
from photo_ocr.detection.craft.postprocessing.polygon import calculate_polygon, PolygonCalculationError


def postprocess(textmap, linkmap, text_threshold, link_threshold, low_text):
    components, link_area = get_word_segmentations(textmap, linkmap, low_text, text_threshold, link_threshold)

    boxes = []
    polygons = []

    for component in components:

        # only want the actual characters, mark space between the letters as background
        #import numpy as np
        #component.segmentation = component.segmentation.astype(np.uint8)
        #component.segmentation[link_area] = 0

        bounding_box = calculate_bounding_box(component, link_area)
        try:
            polygon = calculate_polygon(component.segmentation, bounding_box)
        except PolygonCalculationError:
            polygon = None

        boxes.append(bounding_box)
        polygons.append(polygon)

    return boxes, polygons