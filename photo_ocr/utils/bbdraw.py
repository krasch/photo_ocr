from PIL import Image, ImageDraw, ImageFont


def rectangle(image: Image.Image, coordinates, colour=None, label: str = None, inplace: bool = True) -> Image.Image:
    if not inplace:
        image = image.copy()

    colour = colour or "blue"

    draw = ImageDraw.Draw(image)
    draw.rectangle(coordinates, outline=colour)

    if label:
        text_x, text_y = coordinates[0][0], coordinates[0][1] - 10
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 10)

        width, height = draw.textsize(label, font=fnt)
        draw.rectangle([(text_x, text_y), (text_x + width, text_y + height)], fill="grey")

        draw.text((text_x, text_y), label, font=fnt, fill=colour)

    return image


def polygon(image: Image.Image, coordinates, colour=None, label: str = None, inplace: bool = True) -> Image.Image:
    if not inplace:
        image = image.copy()

    colour = colour or "blue"

    draw = ImageDraw.Draw(image)
    coordinates = coordinates + coordinates[0:1]  # draw.polygon does not support line width
    draw.line(coordinates, fill=colour, width=4)

    # todo label

    return image

