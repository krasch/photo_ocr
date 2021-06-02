import torch.nn as nn

from photo_ocr.recognition.models.modules.original.feature_extraction import VGG_FeatureExtractor, ResNet_FeatureExtractor


class VGG(nn.Module):
    NUM_FEATURES = 512

    def __init__(self, input_shape):
        super(VGG, self).__init__()

        _, _, input_channels = input_shape

        self.model = VGG_FeatureExtractor(input_channel=input_channels, output_channel=self.NUM_FEATURES)
        self.output_size = self.NUM_FEATURES

    def forward(self, batch):
        return self.model.forward(batch)


class ResNet(nn.Module):
    NUM_FEATURES = 512

    def __init__(self, input_shape):
        super(ResNet, self).__init__()

        _, _, input_channels = input_shape

        self.model = ResNet_FeatureExtractor(input_channel=input_channels, output_channel=self.NUM_FEATURES)
        self.output_size = self.NUM_FEATURES

    def forward(self, batch):
        return self.model.forward(batch)