import cv2
import torch
from PIL import Image
import torch.backends.cudnn as cudnn

from photo_ocr.detection.craft.models.craft import craft as load_craft
from photo_ocr.detection.craft.models.refinenet import refine_net as load_refine_net
from photo_ocr.detection.craft.preprocessing import prepare_image
from photo_ocr.detection.craft import postprocessing
from refactoraid import refactoraid

CUDA = torch.cuda.is_available()


class TextDetector:
    def __init__(self):
        self.craft, self.refine_net = self._load_models()

    def detect(self,
               image: Image.Image,
               text_threshold=0.7,
               link_threshold=0.4,
               low_text=0.4,
               refine=True,
               canvas_size=1280,
               mag_ratio=1.5,
               interpolation=cv2.INTER_LINEAR):

        resized, ratio = prepare_image(image, canvas_size, interpolation, mag_ratio)

        #refactoraid.add("image", resized.cpu().data.numpy())

        if CUDA:
            resized = resized.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.craft(resized)

        #refactoraid.add("y", y.cpu().data.numpy())
        #refactoraid.add("feature", feature.cpu().data.numpy())

        # make score and link geo
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if refine:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
            #refactoraid.add("score_link_refined", score_link)

        # Post-processing
        boxes, polys = postprocessing.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
        #refactoraid.add("boxes", boxes)
        #refactoraid.add("polys", polys)

        # coordinate adjustment
        boxes = postprocessing.adjustResultCoordinates(boxes, ratio, ratio)
        polys = postprocessing.adjustResultCoordinates(polys, ratio, ratio)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        return boxes, polys

    @staticmethod
    def _load_models():
        craft = load_craft(pretrained=True, progress=True, freeze=True)
        refine_net = load_refine_net(pretrained=True, progress=True)

        if CUDA:
            craft = craft.cuda()
            refine_net = refine_net.cuda()

            # todo those things necessary?
            craft = torch.nn.DataParallel(craft)
            refine_net = torch.nn.DataParallel(refine_net)
            cudnn.benchmark = False

        craft.eval()
        refine_net.eval()

        return craft, refine_net
