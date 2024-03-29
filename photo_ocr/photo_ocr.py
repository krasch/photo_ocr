from enum import Enum
from typing import Union, List

from PIL import Image

from photo_ocr.typing import OCRResult, RecognitionResult, Polygon
from photo_ocr.detection.detection import Detection
from photo_ocr.recognition.recognition import Recognition
from photo_ocr.util.image import crop_and_align
from photo_ocr.util.batchify import flatten


class InputType(Enum):
    SINGLE_IMAGE = 1
    IMAGE_LIST = 2


class PhotoOCR:
    def __init__(self,
                 detection_params: dict = None,
                 recognition_params: dict = None):
        """
        :param detection_params: Dictionary of parameters for initialising the detection model.
                                 See constructor of :class:`photo_ocr.detection.detection.Detection`
        :param recognition_params: Dictionary of parameters for initialising the recognition model.
                                   See constructor of :class:`photo_ocr.recognition.recognition.Recognition`
        """
        self._detection_init_params = detection_params or {}
        self._recognition_init_params = recognition_params or {}

        # lazy initialisation - the models will be loaded the first time they are needed
        self._detection_model = None
        self._recognition_model = None

    def detection(self,
                  images: Union[Image.Image, List[Image.Image]],
                  ) -> Union[List[Polygon], List[List[Polygon]]]:
        """
        Run the detection algorithm to find text areas in one image or a list of images.

        :param images: One PIL image or a list of PIL images on which to run the detection algorithm.
        :return: If one image was supplied: one detection result (list of text polygons)
                 If list of images was supplied: list of detection results of same length and order as input list.
        """

        # lazy initialisation - load the model if it has not been used before
        if self._detection_model is None:
            self._detection_model = Detection(**self._detection_init_params)

        # check that all inputs are PIL images and identify if we have a single image or a list of images
        input_type = self._validate_image_input(images)

        # run the detection and return the found text polygons
        if input_type == InputType.SINGLE_IMAGE:
            # detection model expects a list of images -> wrap image in list and grab first (only) item from output
            return self._detection_model([images])[0]
        else:
            return self._detection_model(images)

    def recognition(self,
                    images: Union[Image.Image, List[Image.Image]]
                    ) -> Union[RecognitionResult, List[RecognitionResult]]:
        """
        Run text recognition to "read" the word present in the image.

        :param images: One PIL image or a list of PIL images on which to run the recognition algorithm. The image
                       should only contain one word, which should ideally be horizontally aligned.
        :return: If one image was supplied: one recognition result (tuple of (word, confidence))
                 If list of images was supplied: list of recognition results of same length and order as input list.
        """

        # lazy initialisation - load the model if it has not been used before
        if self._recognition_model is None:
            self._recognition_model = Recognition(**self._recognition_init_params)

        # check that all inputs are PIL images and identify if we have a single image or a list of images
        input_type = self._validate_image_input(images)

        # run the recognition and return word(s) and confidence(s)
        if input_type == InputType.SINGLE_IMAGE:
            # detection model expects a list of images -> wrap image in list and grab first (only) item from output
            return self._recognition_model([images])[0]
        else:
            return self._recognition_model(images)

    def ocr(self,
            images: Union[Image.Image, List[Image.Image]],
            confidence_threshold: float = 0.2,
            ) -> Union[List[OCRResult], List[List[OCRResult]]]:
        """

        :param images: One PIL image or a list of PIL images on which to run OCR.
        :param confidence_threshold: Only recognitions with confidence larger than this threshold will be returned.
        :return: If one image was supplied: list of OCR results (tuple of polygon, word, confidence)
                 If list of images was supplied: list of (list of OCR results) of same length and order as input list
        """

        # check that all inputs are PIL images and identify if we have a single image or a list of images
        input_type = self._validate_image_input(images)

        # run the ocr
        if input_type == InputType.SINGLE_IMAGE:
            return self._run_ocr([images], confidence_threshold)[0]
        else:
            return self._run_ocr(images, confidence_threshold)

    def _run_ocr(self,
                 images: List[Image.Image],
                 confidence_threshold: float) -> List[List[OCRResult]]:

        # for all images: get bounding polygons of all words in the image
        polygons = self.detection(images)

        # cut out each of the found words and align horizontally
        crops = [crop_and_align(image, polygons_for_image) for image, polygons_for_image in zip(images, polygons)]

        # to benefit from batch processing in text recognition, take the nested list of polygons per image
        # and turn it into a flat list of all polygons; unflatten function allows us to reverse this later
        crops, unflatten = flatten(crops)

        # run recognition on each of the cropped images, returns for each crop a tuple of (word, confidence)
        recognitions = self.recognition(crops)

        # turn flat list of recognitions over all images into a nested list of recognitions per image
        recognitions = unflatten(recognitions)

        # for each image, wrap detection results and recognition results together in ocr result tuples,
        results = [[OCRResult(polygon, rec.text, rec.confidence) for polygon, rec in zip(*results_for_image)]
                   for results_for_image in zip(polygons, recognitions)]

        # for each image, keep only the high-confidence words
        results = [[result for result in results_for_image if result.confidence >= confidence_threshold]
                   for results_for_image in results]

        # for each image,  most confident results come first
        results = [sorted(results_for_image, key=lambda item: item.confidence, reverse=True)
                   for results_for_image in results]

        return results

    @staticmethod
    def _validate_image_input(images):
        # looks like input is a single image
        if isinstance(images, Image.Image):
            return InputType.SINGLE_IMAGE

        # looks like input is a list of images
        elif isinstance(images, List):
            all_entries_are_images = all([isinstance(image, Image.Image) for image in images])

            if all_entries_are_images:
                return InputType.IMAGE_LIST
            else:
                msg = "At least one entry in your list of images is not a PIL image." \
                      "You must pass either a single PIL image or a list of PIL images"
                raise TypeError(msg)

        # looks like input is something else
        else:
            msg = "Objects of type {} are not supported." \
                  "You must pass either a single PIL image or a list of PIL images"
            raise TypeError(msg.format(type(images)))


# instantiate PhotoOCR with the default settings and make some of its methods available globally
# IMPORTANT: PhotoOCR does lazy loading, i.e. this does not load the models, instead they will be loaded at first use
_ocr = PhotoOCR()
ocr = _ocr.ocr  # perform both detection and recognition
detection = _ocr.detection  # perform only detection
recognition = _ocr.recognition  # perform only recognition
