from typing import List

from PIL import Image

from photo_ocr.typing import Polygon
from photo_ocr.detection.model_zoo import craft
from photo_ocr.detection.craft.preprocessing import calculate_resize_ratio, init_transforms
from photo_ocr.detection.craft.postprocessing import init_postprocessing
from photo_ocr.util.cuda import DEVICE


class Detection:
    def __init__(self,
                 # all default parameter values taken from original code
                 image_max_size: int = 1280,
                 image_magnification: int = 1.5,
                 combine_words_to_lines: bool = False,
                 text_threshold_first_pass: float = 0.4,
                 text_threshold_second_pass: float = 0.7,
                 link_threshold: float = 0.4
                 ):
        """
        Convenience class for text detection.
        Combines the image preprocessing, model and result postprocessing steps.
        Currently only supports the CRAFT text detection method.

        :param image_max_size:
                    During image pre-processing before running text detection, the image will be resized such that the
                    larger side of the image is smaller than image_max_size.
        :param image_magnification:
                    During image pre-processing before running text detection, the image will be magnified by this
                    value (but no bigger than image_max_size); should be >= 1.0
        :param combine_words_to_lines:
                    If true, use the additional "RefineNet" to link individual words
                    that are near each other horizontally together.
        :param text_threshold_first_pass:
                    The CRAFT model produces for every pixel a score of how
                    likely it is that this pixel is part of a text character (called regions score in the paper).
                    During postprocessing, only those pixels are considered, that are above the
                    text_threshold_first_pass. Value must be in [0.0, 1.0].
        :param text_threshold_second_pass:
                    See explanation of text_threshold_first_pass.
                    During postprocessing, there is a second round of thresholding happening
                    after the individual characters have been linked together to words
                    (see link_threshold). Value must be in [0.0, 1.0];
                    detection_text_threshold_second_pass <= detection_text_threshold_first_pass
        :param link_threshold:
                    The CRAFT model produces for every pixels a score of how likely it is that
                    this pixel is between two text characters (called affinity score in the paper). During
                    postprocessing, this score is used to link individual characters together as words.
                    Only pixels that are above the link_threshold are considered. Value must be in [0.0, 1.0]
        """
        self.model = craft(pretrained=True, progress=True)
        self.model.eval()

        self.image_max_size = image_max_size
        self.image_magnification = image_magnification
        self.combine_words_to_lines = combine_words_to_lines
        self.postprocess = init_postprocessing(text_threshold_first_pass, text_threshold_second_pass, link_threshold)

    def __call__(self,
                 images: List[Image.Image],
                 ) -> List[List[Polygon]]:
        """
        Run text detection on the given images.

        :param images: images on which to run the detection
        :return: One detection result per input image. Each detection result is a list of all found text polygons.
        """

        # currently no batching implemented, so just run all images one after the other.
        # reason for no batching: worried of overhead since all images would need to be the same size
        # todo consider a parameter batch_mode that user can set if they want to run in batches
        return [self._detect_one_image(image) for image in images]

    def _detect_one_image(self, image: Image.Image) -> List[Polygon]:
        """
        Helper function that runs the text detection on one image.
        :param image:
        :return:
        """

        # perform image preprocessing
        image = image.convert("RGB")  # in case we got a grayscale or transparent image
        resize_ratio = calculate_resize_ratio(image, self.image_max_size, self.image_magnification)
        image = init_transforms(resize_ratio)(image)

        # forward pass
        # score_text = for each pixel, how likely is it that this pixel is part of a text character
        # link_text = for each pixel, how likely it is that this pixel is between two text characters
        batch = image.unsqueeze(0).to(DEVICE)
        score_text, score_link = self.model(batch, refine=self.combine_words_to_lines)

        # only have one image, so just grab the first result
        score_text = score_text[0, :, :].cpu().data.numpy()
        score_link = score_link[0, :, :].cpu().data.numpy()

        # for each detected word, calculate first a bounding box and then a tight polygon around the word
        polygons = self.postprocess(score_text, score_link)

        # detections are based on the resized image -> need to map them back to the original image
        ratio = 1.0 / resize_ratio  # to reverse the initial resizing
        ratio = ratio * 2.0  # the craft net produces segmentation maps that are half the size of the input image

        # perform the adjustment with the ratio just calculated
        polygons = [polygon * ratio for polygon in polygons]

        # convert to PIL polygon format (list of (x, y) tuples)
        polygons = [[(x, y) for x, y in polygon] for polygon in polygons]

        return polygons
