from collections import OrderedDict

from torchvision.models.utils import load_state_dict_from_url

from photo_ocr.util.cuda import DEVICE
from photo_ocr.detection.craft.model import CraftTextDetectionModel


model_urls = {
    "craft": "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
    "refine_net": "https://drive.google.com/uc?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO",
}


def craft(pretrained, progress):

    # remove "module." from beginning of all layer names
    def rename_layers(weights):
        weights = OrderedDict([(layer_name[7:], layer_weights) for layer_name, layer_weights in weights.items()])
        return weights

    model = CraftTextDetectionModel()

    # need to provide filename because otherwise files from google drive are all stored under the same name
    if pretrained:
        # load the weights for the main craft model
        state_dict = load_state_dict_from_url(model_urls['craft'],
                                              file_name="craft_mlt_25k.pth",
                                              progress=progress, map_location=DEVICE)
        state_dict = rename_layers(state_dict)
        model.craft.load_state_dict(state_dict)

        # load the weights for the additional refiner model
        state_dict = load_state_dict_from_url(model_urls['refine_net'],
                                              file_name="craft_refiner_CTW1500.pth",
                                              progress=progress, map_location=DEVICE)
        state_dict = rename_layers(state_dict)
        model.refiner.load_state_dict(state_dict)

    return model



