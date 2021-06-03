from collections import OrderedDict

from torchvision.models.utils import load_state_dict_from_url

from photo_ocr.cuda import DEVICE
from photo_ocr.recognition.models.base import TextRecognitionModel
from photo_ocr.recognition.models.modules.wrappers import transformation, feature_extraction, sequence_modeling, prediction

CHARACTERS = list("0123456789abcdefghijklmnopqrstuvwxyz")
CHARACTERS_CASE_SENSITIVE = CHARACTERS + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~")


model_urls = {
    "None_VGG_None_CTC": "https://www.dropbox.com/sh/j3xmli4di1zuv3s/AACpKykhWSRBUU7xl2LGgt9ja/None-VGG-None-CTC.pth?dl=1",
    "None_ResNet_None_CTC": "https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth?dl=1",
    "TPS_ResNet_BiLSTM_Attn": "https://www.dropbox.com/sh/j3xmli4di1zuv3s/AADbTu4LF-nMUBmC43_RQ8OGa/TPS-ResNet-BiLSTM-Attn.pth?dl=1",
    "TPS_ResNet_BiLSTM_Attn_case_sensitive": "https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABIa4z2u5InghKfUpiLREZKa/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth?dl=1",
    "TPS_ResNet_BiLSTM_CTC": "https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAB0X-sX05-0psb4uXWPYSmza/TPS-ResNet-BiLSTM-CTC.pth?dl=1"
}


# during refactoring some classes were moved -> layer names changed
def _rename_layers(weights):
    map_ = {"module.Transformation": "transformation.model",
            "module.FeatureExtraction": "feature_extraction.model",
            "module.SequenceModeling": "sequence_modeling.model",
            "module.Prediction": "prediction.model"}

    def rename_layer(layer):
        # layer consists of name of the stage followed by the actual layer name; split those two up
        # e.g. module.FeatureExtraction.ConvNet.bn0_1.bias
        # -> stage=module.FeatureExtraction     layer=ConvNet.bn0_1.bias
        stage = ".".join(layer.split(".")[:2])
        layer = ".".join(layer.split(".")[2:])

        # only the stage part needs to be updated
        stage = map_[stage]

        # put it all back together
        return stage + "." + layer

    weights = OrderedDict([(rename_layer(layer_name), layer_weights) for layer_name, layer_weights in weights.items()])
    return weights


def _load_weights(model, url, progress):
    state_dict = load_state_dict_from_url(url, progress=progress, map_location=DEVICE)
    state_dict = _rename_layers(state_dict)
    model.load_state_dict(state_dict)


def None_VGG_None_CTC(image_shape, pretrained, progress):
    # config
    stages = {"transformation": transformation.NoTransformation,
              "feature_extraction": feature_extraction.VGG,
              "sequence_modeling": sequence_modeling.NoSequenceModel,
              "prediction": prediction.CTC}
    characters = CHARACTERS

    # initialisation
    model = TextRecognitionModel(image_shape, characters, stages)
    if pretrained:
        _load_weights(model, model_urls['None_VGG_None_CTC'], progress)

    return model


def None_ResNet_None_CTC(image_shape, pretrained, progress):
    # config
    stages = {"transformation": transformation.NoTransformation,
              "feature_extraction": feature_extraction.ResNet,
              "sequence_modeling": sequence_modeling.NoSequenceModel,
              "prediction": prediction.CTC}
    characters = CHARACTERS

    # initialisation
    model = TextRecognitionModel(image_shape, characters, stages)
    if pretrained:
        _load_weights(model, model_urls['None_ResNet_None_CTC'], progress)

    return model


def TPS_ResNet_BiLSTM_Attn(image_shape, pretrained, progress):
    # config
    stages = {"transformation": transformation.TPS,
              "feature_extraction": feature_extraction.ResNet,
              "sequence_modeling": sequence_modeling.BiLSTM,
              "prediction": prediction.Attention}
    characters = CHARACTERS

    # initialisation
    model = TextRecognitionModel(image_shape, characters, stages)
    if pretrained:
        _load_weights(model, model_urls['TPS_ResNet_BiLSTM_Attn'], progress)

    return model


def TPS_ResNet_BiLSTM_Attn_case_sensitive(image_shape, pretrained, progress):
    # config
    stages = {"transformation": transformation.TPS,
              "feature_extraction": feature_extraction.ResNet,
              "sequence_modeling": sequence_modeling.BiLSTM,
              "prediction": prediction.Attention}
    characters = CHARACTERS_CASE_SENSITIVE

    # initialisation
    model = TextRecognitionModel(image_shape, characters, stages)
    if pretrained:
        _load_weights(model, model_urls['TPS_ResNet_BiLSTM_Attn_case_sensitive'], progress)

    return model


def TPS_ResNet_BiLSTM_CTC(image_shape, pretrained, progress):
    # config
    stages = {"transformation": transformation.TPS,
              "feature_extraction": feature_extraction.ResNet,
              "sequence_modeling": sequence_modeling.BiLSTM,
              "prediction": prediction.CTC}
    characters = CHARACTERS

    # initialisation
    model = TextRecognitionModel(image_shape, characters, stages)
    if pretrained:
        _load_weights(model, model_urls['TPS_ResNet_BiLSTM_CTC'], progress)

    return model
