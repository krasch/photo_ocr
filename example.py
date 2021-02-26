from photo_ocr import load_image, save_image, draw_polygons, perform_ocr

# load the image
image = load_image("examples/rifles.png")

# run the ocr
text_bounding_boxes, text_polygons = perform_ocr(image)

# draw results on image and save
image_with_polygons = draw_polygons(image, text_polygons)
save_image(image, "examples/detections.png")

