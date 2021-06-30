from photo_ocr import load_image, ocr, detection, recognition, draw_ocr, draw_detections

# you can also use PIL Image.open, but load_image automatically rotates the image according to the EXIF metadata!
# (download pub.jpg here: https://github.com/krasch/photo_ocr/blob/master/pub.jpg)
image = load_image("pub.jpg")

# run the ocr
results = ocr(image)

# loop over the results
for result in results:
    # polygon around the text
    # (list of xy coordinates: [(x0, y0), (x1, y1),  ....])
    print(result.polygon)

    # the actual text (a string)
    print(result.text)

    # the recognition confidence (a number in [0.0, 1.0])
    print(result.confidence)

# an alternative way to loop over the results
for polygon, text, confidence in results:
    print(polygon)
    print(text)
    print(confidence)

# draw bounding polygons and text on the image
image = draw_ocr(image, results)

# done!
image.save("pub_ocr.jpg")

# use the `detection` function to only run the text detection step
image = load_image("pub.jpg")
polygons = detection(image)

# list of polygons where text was found
for polygon in polygons:
    # polygon around the text
    # (list of xy coordinates: [(x0, y0), (x1, y1),  ....])
    print(polygon)


# draw_detections draws detection results
image = draw_detections(image, polygons)
image.save("pub_detections.jpg")

# use the `recognition` function to only run the text recognition step
# the input image must have been cropped to text polygon, text should be aligned horizontally
# (download crop.jpg here: https://github.com/krasch/photo_ocr/blob/master/crop.jpg)
image = load_image("crop.jpg")
text, confidence = recognition(image)
print(text, confidence)



