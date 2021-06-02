from torch import nn as nn
from torch.nn import functional as F


class TextRecognitionModel(nn.Module):

    def __init__(self, image_shape, characters, stages):
        super(TextRecognitionModel, self).__init__()

        self.transformation = stages["transformation"](image_shape)
        self.feature_extraction = stages["feature_extraction"](image_shape)
        self.pooling = nn.AdaptiveAvgPool2d((None, 1))
        self.sequence_modeling = stages["sequence_modeling"](self.feature_extraction.output_size)
        self.prediction = stages["prediction"](self.sequence_modeling.output_size, characters)

    def forward(self, batch):
        # transform image so text is more rectangular (optional)
        # input shape: [batch_size, num_channels=1, image_width, image_height]
        # output shape: same as input shape
        batch = self.transformation(batch)

        # extract visual features
        # output shape: [batch_size, features_dim1=512, num_channels=1, features_dim2] (features_dim2 depends on model)
        features = self.feature_extraction(batch)

        # output shape: [batch_size, features_dim2, features_dim1=512, num_channels=1]
        features = features.permute(0, 3, 1, 2)

        # adaptive avg pooling
        # output shape: same as input shape
        features = self.pooling(features)

        # get rid of channel dimension
        # output shape: [batch_size, features_dim2, features_dim1]
        features = features.squeeze(3)

        # capture some contextual information between the features (optional)
        # (because we have a sequence of characters, not independent characters)
        # output shape: [batch_size, new_features_dim2, _new_features_dim1] (dim1, dim2 depend on model)
        features = self.sequence_modeling(features)

        # prediction character classes, each class corresponds to a character
        # output shape: [batch_size, max_word_length, num_character_classes]
        prediction = self.prediction(features)

        # softmax over each position in word
        return F.softmax(prediction, dim=2)

    def decode(self, predictions):
        """
        :param predictions: as returned by .forward method
        :return: predicted text, text_confidence
        """

        # at each position, what is the most likely character class and what is the probability of this class?
        character_indexes = predictions.argmax(1)
        character_probabilities = predictions.max(1)

        # map the character classes to actual characters (depends on prediction model used)
        characters = self.prediction.decode(character_indexes)

        # no valid character was found
        if len(characters) == 0:
            confidence_score = 0.0
            return characters, confidence_score

        # calculate the cumulative confidence, todo this is probably wrong with CTC, also in original code? (yes, also in original code)
        character_probabilities = character_probabilities[0:len(characters)]
        confidence_score = character_probabilities.cumprod()[-1]

        return characters, confidence_score