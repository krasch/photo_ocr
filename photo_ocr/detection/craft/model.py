from torch import nn as nn

from photo_ocr.detection.craft.modules.original.refinenet import RefineNet
from photo_ocr.detection.craft.modules.original.craft import CRAFT


class CraftTextDetectionModel(nn.Module):
    def __init__(self):
        super(CraftTextDetectionModel, self).__init__()

        self.craft = CRAFT(freeze=True, pretrained=False)
        self.refiner = RefineNet()

    def forward(self, batch, refine=False):
        y, feature = self.craft(batch)

        score_text = y[:, :, :, 0]
        score_link = y[:, :, :, 1]

        if refine:
            y_refiner = self.refiner(y, feature)
            score_link = y_refiner[:, :, :, 0]

        return score_text, score_link

    def eval(self):
        self.craft.eval()
        self.refiner.eval()







