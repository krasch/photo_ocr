from typing import Callable, List

import torch
from PIL import Image

from photo_ocr.typing import RecognitionResult
from photo_ocr.recognition.model_zoo import TPS_ResNet_BiLSTM_Attn
from photo_ocr.recognition.models.preprocessing import init_transforms
from photo_ocr.util.batchify import run_in_batches


class Recognition:
    def __init__(self,
                 model: Callable = TPS_ResNet_BiLSTM_Attn,
                 image_width:  int = 100,
                 image_height: int = 32,
                 keep_ratio:   bool = False,
                 batch_size: int = 32):
        """

        Convenience class for text recognition.
        Combines the image preprocessing, model and result postprocessing steps.
        Currently only supports the CRAFT text detection method.

        :param model: One of the initialisation functions in the photo_ocr.recognition.model_zoo
        :param image_width: During image pre-processing, the (cropped) image will be resized to this width
                            models were trained with width=100, other values don't seem to work as well
        :param image_height: During image pre-processing, the (cropped) image will be resized to this height;
                             models were trained with height=32, other values don't seem to work as well
        :param keep_ratio:  When resizing images during pre-processing: True -> keep the width/height
                            ratio (and pad appropriately) or False -> simple resize without keeping ratio
        :param batch_size: Size of the batches to be fed to the model.
        """

        # grayscale, resizing, etc
        image_shape = (image_height, image_width, 1)
        self.preprocess = init_transforms(image_shape, keep_ratio)

        # load the actual model
        self.model = model(image_shape=image_shape, pretrained=True, progress=True)
        self.model.eval()

        # converting prediction scores to predicted characters
        self.postprocess = self.model.decode

        self.batch_size = batch_size

    def __call__(self, images: List[Image.Image]) -> List[RecognitionResult]:
        """

        Run the text recognition model on the given images.

        :param images:
        :return: List of recognition results, same length and order as input array. Each recognition result is
                a tuple of (word, confidence)
        """
        # apply conversion to grayscale, resizing, etc
        images = [self.preprocess(image) for image in images]

        # run the prediction, avoid memory errors by doing that in batches
        # run_in_batches is a generator function -> wrap in list
        with torch.no_grad():
            scores = list(run_in_batches(self.model, images, batch_size=self.batch_size))

        # convert the predicted scores into the actual characters
        results = [self.postprocess(pred) for pred in scores]

        # wrap the results into convenient named tuples for easier access
        results = [RecognitionResult(text, confidence) for text, confidence in results]

        return results
