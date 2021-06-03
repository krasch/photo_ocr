# todo sort this one out
from typing import List
from bbdraw.types import Rectangle, LabeledRectangle

from PIL import Image, ImageDraw, ImageFont


def _label(draw, location, label_text, colour):
    text_x, text_y = location
    fnt = ImageFont.truetype("Pillow/Tests/fonts/DejaVuSans/DejaVuSans.ttf", 40)

    width, height = draw.textsize(label_text, font=fnt)
    draw.rectangle([(text_x, text_y), (text_x + width, text_y + height+5)], fill=colour)

    draw.text((text_x, text_y), label_text, font=fnt, fill="white")


def rectangle(image: Image.Image, rect: Rectangle, label: str = None, colour=None) -> Image.Image:
    colour = colour or "blue"

    draw = ImageDraw.Draw(image)
    draw.rectangle(rect, outline=colour)

    if label:
        text_x, text_y = rect[0][0], rect[0][1] - 10
        _label(draw, (text_x, text_y), label, colour)

    return image


def rectangles(image: Image.Image, labeled_rects: List[LabeledRectangle], colour=None) -> Image.Image:
    for rect, label in labeled_rects:
        image = rectangle(image, rect, label, colour=colour)
    return image


class BBDraw:
    def __init__(self, image):
        self.image = image

    def rectangle(self, rect: Rectangle, label: str = None, colour=None) -> None:
        self.image = rectangle(self.image, rect, label, colour)

    def rectangles(self, labeled_rects: List[LabeledRectangle], colour=None) -> None:
        self.image = rectangles(self.image, labeled_rects, colour)




def polygon(image: Image.Image, coordinates, colour=None, label: str = None, inplace: bool = True) -> Image.Image:
    if not inplace:
        image = image.copy()

    colour = colour or "blue"

    draw = ImageDraw.Draw(image)
    coordinates = coordinates + coordinates[0:1]  # draw.polygon does not support line width
    draw.line(coordinates, fill=colour, width=4)

    if label:
        text_x, text_y = coordinates[0][0], coordinates[0][1] - 10
        _label(draw, (text_x, text_y), label, colour)

    return image

