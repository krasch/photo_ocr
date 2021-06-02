import torch.nn as nn

from photo_ocr.recognition.models.modules.original.sequence_modeling import BidirectionalLSTM


class BiLSTM(nn.Module):
    NUM_HIDDEN_UNITS = 256

    def __init__(self, input_size):
        super(BiLSTM, self).__init__()

        self.model = nn.Sequential(
            BidirectionalLSTM(input_size, self.NUM_HIDDEN_UNITS, self.NUM_HIDDEN_UNITS),
            BidirectionalLSTM(self.NUM_HIDDEN_UNITS, self.NUM_HIDDEN_UNITS, self.NUM_HIDDEN_UNITS))
        self.output_size = self.NUM_HIDDEN_UNITS

    def forward(self, batch):
        return self.model.forward(batch)


class NoSequenceModel(nn.Module):

    def __init__(self, input_size):
        super(NoSequenceModel, self).__init__()
        self.output_size = input_size

    def forward(self, batch):
        return batch

