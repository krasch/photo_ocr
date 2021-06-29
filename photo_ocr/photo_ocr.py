from enum import Enum
from typing import Union, List, NamedTuple

from PIL import Image

from photo_ocr.detection.detection import Detection, Polygon
from photo_ocr.recognition.recognition import Recognition, RecognitionResult
from photo_ocr.util.image import crop_and_align


class InputType(Enum):
    SINGLE_IMAGE = 1
    IMAGE_LIST = 2


OCRResult = NamedTuple("OCRResult", [("polygon", Polygon),
                                     ("text", str),
                                     ("confidence", float)])


class PhotoOCR:
    def __init__(self, recognition_params: dict = None):
        """
        :param recognition_params: Dictionary of parameters for initialising the recognition model.
                                   See constructor of :class:`photo_ocr.recognition.recognition.Recognition`
                                   (text detection currently does not take init params, so there is
                                    no need for detection_init_params)
        """

        self._recognition_init_params = recognition_params or {}

        # lazy initialisation - the models will be loaded the first time they are needed
        self._detection_model = None
        self._recognition_model = None

    def detection(self,
                  images: Union[Image.Image, List[Image.Image]],
                  detection_params: dict = None,
                  ) -> Union[List[Polygon], List[List[Polygon]]]:
        """
        Run the detection algorithm to find text areas in one image or a list of images.

        :param images: One PIL image or a list of PIL images on which to run the detection algorithm.
        :param detection_params: Dictionary of parameters to pass to the detection model,
               See __call__ method of :class:`photo_ocr.detection.detection.Detection`
        :return: If one image was supplied: one detection result (list of text polygons)
                 If list of images was supplied: list of detection results of same length and order as input list.
        """
        detection_params = detection_params or {}

        # lazy initialisation - load the model if it has not been used before
        if self._detection_model is None:
            self._detection_model = Detection()

        # check that all inputs are PIL images and identify if we have a single image or a list of images
        input_type = self._validate_image_input(images)

        # run the detection and return the found text polygons
        if input_type == InputType.SINGLE_IMAGE:
            # detection model expects a list of images -> wrap image in list and grab first (only) item from output
            return self._detection_model([images], **detection_params)[0]
        else:
            return self._detection_model(images, **detection_params)

    def recognition(self, images: Union[Image.Image, List[Image.Image]]) \
            -> Union[RecognitionResult, List[RecognitionResult]]:
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
            detection_params: dict = None,
            confidence_threshold: float = 0.2,
            ) -> Union[List[OCRResult], List[List[OCRResult]]]:
        """

        :param images: One PIL image or a list of PIL images on which to run OCR.
        :param detection_params: Dictionary of parameters to pass to the detection model,
               See __call__ method of :class:`photo_ocr.detection.detection.Detection`
               (recognition currently does not take any parameters, so no need for recognition_params)
        :param confidence_threshold: Only recognitions with confidence larger than this threshold will be returned.
        :return: If one image was supplied: list of OCR results (tuple of polygon, word, confidence)
                 If list of images was supplied: list of (list of OCR results) of same length and order as input list
        """

        # check that all inputs are PIL images and identify if we have a single image or a list of images
        input_type = self._validate_image_input(images)

        # run the ocr
        if input_type == InputType.SINGLE_IMAGE:
            return self._ocr_one(images, detection_params, confidence_threshold)
        else:
            return [self._ocr_one(image, detection_params, confidence_threshold) for image in images]

    def _ocr_one(self,
                 image: Image.Image,
                 detection_params: dict,
                 confidence_threshold: float) -> List[OCRResult]:

        # get bounding polygons of all words in the image
        text_polygons = self.detection(image, detection_params)

        # cut out each of the found words and align horizontally
        crops = [crop_and_align(image, polygon) for polygon in text_polygons]

        # run recognition on each of the cropped images, returns for each image a tuple of (word, confidence)
        recognitions = [self.recognition(crop) for crop in crops]

        # put detection results and recognition results together in a ocr result tuple
        results = [OCRResult(polygon, rec.text, rec.confidence) for polygon, rec in zip(text_polygons, recognitions)]

        # keep only the high-confidence words
        results = [result for result in results if result.confidence >= confidence_threshold]

        # most confident results come first
        results = sorted(results, key=lambda item: item.confidence, reverse=True)

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
