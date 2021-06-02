from torch import LongTensor
import torch.nn as nn

from photo_ocr.recognition.models.modules.original.prediction import Attention as AttentionModule


class CTC(nn.Module):
    def __init__(self, input_size, characters):
        super(CTC, self).__init__()

        # dummy '[CTCblank]' token for CTCLoss (index 0)
        self.characters = ["[CTCblank]"] + characters

        # initialise the model
        self.model = nn.Linear(input_size, len(self.characters))

    def forward(self, batch):
        return self.model.forward(batch.contiguous())

    def decode(self, predictions):
        # todo this is based on original code, not sure about the interactions here, should unit test before refactoring
        def remove_blank_token_and_repeated_characters(text_index):
            for i in range(len(text_index)):
                if text_index[i] != "[CTCblank]" and (not (i > 0 and text_index[i - 1] == text_index[i])):
                    yield text_index[i]

        # map from character class to actual character
        text = [self.characters[i] for i in predictions]

        # do some cleaning
        text = list(remove_blank_token_and_repeated_characters(text))

        return "".join(text)


class Attention(nn.Module):
    HIDDEN_SIZE = 256
    BATCH_MAX_LENGTH = 25  # todo should this be configurable?

    def __init__(self, input_size, characters):
        super(Attention, self).__init__()

        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.characters = ['[GO]', '[s]'] + characters

        # initialise the model
        self.model = AttentionModule(input_size, self.HIDDEN_SIZE, len(self.characters))

    def forward(self, batch):
        batch_size = batch.shape[0]
        placeholder = LongTensor(batch_size, self.BATCH_MAX_LENGTH + 1).fill_(0)
        return self.model(batch.contiguous(), placeholder, is_train=False, batch_max_length=self.BATCH_MAX_LENGTH)

    def decode(self, predictions):
        # map from character class to actual character
        text = ''.join([self.characters[i] for i in predictions])

        # prune after "end of sentence" token ([s])
        return text[:text.find('[s]')]
