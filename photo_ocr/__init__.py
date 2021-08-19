from photo_ocr.photo_ocr import ocr, detection, recognition, PhotoOCR
from photo_ocr.util.image import load_image, draw_ocr, draw_detections
from photo_ocr.util.config import config as photo_ocr_config

# this warning shows up when using torchvision==0.10.0
# it is caused internally by torch==1.9.0 and will be fixed in the next pytorch release
# see https://github.com/pytorch/pytorch/issues/54846
# until then: suppress that specific warning so that user is not confused
import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        message="Named tensors and all their associated APIs are an experimental feature")
