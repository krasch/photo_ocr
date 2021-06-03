from torchvision.models.utils import load_state_dict_from_url

from photo_ocr.detection.craft.model import CraftTextDetectionModel

# 1. These weights use different layer names than the original ones. Why?
# The original repository contains links to weights in a google drive.
# It seems like the original weights were saved with different layer names than the final code, so loading
# the weights into a model, it was first necessary to rename some layers.
# To avoid all this extra code, stored the weights with the correct names.
# 2. How the weights were stored
# _use_new_zipfile_serialization=True because load_state_dict_from_url broken with new zipfile format
# 3. TL;DR:
# -> these weights are exactly the same as the original ones, just use the correct layer names
model_urls = {
    "craft": "https://photoocr.s3-eu-west-1.amazonaws.com/craft_mlt_25k.pth",
    "refine_net": "https://photoocr.s3-eu-west-1.amazonaws.com/craft_refiner_CTW1500.pth",
}


def craft(pretrained, progress):
    model = CraftTextDetectionModel()

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['craft'], progress=progress)
        model.craft.load_state_dict(state_dict)

        state_dict = load_state_dict_from_url(model_urls['refine_net'], progress=progress)
        model.refiner.load_state_dict(state_dict)

    return model



