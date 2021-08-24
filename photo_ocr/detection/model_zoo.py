from collections import OrderedDict

from torchvision.models.utils import load_state_dict_from_url

from photo_ocr.util.cuda import DEVICE
from photo_ocr.detection.craft.model import CraftTextDetectionModel

# this github repository re-hosts the pre-trained model open sourced by Clova AI Research / NAVER Corp
# see README.md#section-licensing and LICENSE_DETECTION.txt
BASE_URL = "https://github.com/krasch/photo_ocr_models/releases/download/text-detection-models-20190928/"

# pytorch saves model files in a flat file structure
# -> use this filename prefix to make model files easily recognizable and give some version info
PREFIX = "text-detection-20190928__"

model_urls = {
    "craft": BASE_URL + PREFIX + "craft_mlt_25k-4a5efbfb.pth",
    "refine_net": BASE_URL + PREFIX + "craft_refiner_CTW1500-f7000cd3.pth",
}


def craft(pretrained, progress):

    # remove "module." from beginning of all layer names
    def rename_layers(weights):
        weights = OrderedDict([(layer_name[7:], layer_weights) for layer_name, layer_weights in weights.items()])
        return weights

    model = CraftTextDetectionModel()

    if pretrained:
        # load the weights for the main craft model
        state_dict = load_state_dict_from_url(model_urls['craft'], progress=progress, map_location=DEVICE,
                                              check_hash=True)
        state_dict = rename_layers(state_dict)
        model.craft.load_state_dict(state_dict)

        # load the weights for the additional refiner model
        state_dict = load_state_dict_from_url(model_urls['refine_net'], progress=progress, map_location=DEVICE,
                                              check_hash=True)
        state_dict = rename_layers(state_dict)
        model.refiner.load_state_dict(state_dict)

    model.craft = model.craft.to(DEVICE)
    model.refiner = model.refiner.to(DEVICE)

    return model



