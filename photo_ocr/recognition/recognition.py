import torch


from photo_ocr.recognition.preprocessing import init_transforms
from photo_ocr.util.batchify import run_in_batches


class Recognition:
    def __init__(self, model_class, image_shape, keep_ratio): # todo model class is a bad name
        # grayscale, resizing, etc
        self.preprocess = init_transforms(image_shape, keep_ratio)

        # load the actual model
        self.model = model_class(image_shape=image_shape, pretrained=True, progress=True)
        self.model.eval()

        # converting prediction scores to predicted characters
        self.postprocess = self.model.decode

    def run_one(self, image):
        images = [image]
        return self.run_all(images)[0]

    def run_all(self, images):
        # apply conversion to grayscale, resizing, etc
        images = [self.preprocess(image) for image in images]

        # run the prediction, avoid memory errors by doing that in batches
        # run_in_batches is a generator function -> wrap in list
        with torch.no_grad():
            scores = list(run_in_batches(self.model, images, batch_size=2))

        # convert the predicted scores into the actual characters
        predictions = [self.postprocess(pred) for pred in scores]

        return predictions


