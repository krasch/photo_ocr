from collections import OrderedDict

from torchvision.models.utils import load_state_dict_from_url

from photo_ocr.util.config import config
from photo_ocr.detection.craft.model import CraftTextDetectionModel

# this github repository re-hosts the pre-trained model open sourced by Clova AI Research / NAVER Corp
# see README.md#section-licensing and LICENSE_DETECTION.txt
BASE_URL = "https://github.com/krasch/photo_ocr_models/releases/download/text-detection-models-20190928/"


model_urls = {
    "craft": BASE_URL + "craft_mlt_25k.pth",
    "refine_net": BASE_URL + "craft_refiner_CTW1500.pth",
}


def craft(pretrained, progress):

    # remove "module." from beginning of all layer names
    def rename_layers(weights):
        weights = OrderedDict([(layer_name[7:], layer_weights) for layer_name, layer_weights in weights.items()])
        return weights

    model = CraftTextDetectionModel()
    device = config.get_device()

    if pretrained:
        # load the weights for the main craft model
        state_dict = load_state_dict_from_url(model_urls['craft'], progress=progress, map_location=device)
        state_dict = rename_layers(state_dict)
        model.craft.load_state_dict(state_dict)

        # load the weights for the additional refiner model
        state_dict = load_state_dict_from_url(model_urls['refine_net'], progress=progress, map_location=device)
        state_dict = rename_layers(state_dict)
        model.refiner.load_state_dict(state_dict)

    model.craft = model.craft.to(device)
    model.refiner = model.refiner.to(device)

    return model



