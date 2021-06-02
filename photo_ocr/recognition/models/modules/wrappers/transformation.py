import torch.nn as nn

from photo_ocr.recognition.models.modules.original.transformation import TPS_SpatialTransformerNetwork


class TPS(nn.Module):
    NUM_FIDUCIAL_POINTS = 20

    def __init__(self, input_shape):
        super(TPS, self).__init__()

        height, width, num_channels = input_shape
        self.model = TPS_SpatialTransformerNetwork(F=self.NUM_FIDUCIAL_POINTS,
                                                   I_size=(height, width),
                                                   I_r_size=(height, width),
                                                   I_channel_num=num_channels)

    def forward(self, batch):
        return self.model.forward(batch)


class NoTransformation(nn.Module):
    def __init__(self, input_shape):
        super(NoTransformation, self).__init__()

    def forward(self, batch):
        return batch

