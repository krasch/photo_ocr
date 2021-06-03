
from photo_ocr import load_image, save_image, ocr, draw_ocr_results

# load the image
image = load_image("examples/rifles.png")

# run the ocr
results = ocr(image)

# draw the results on the image and save image
draw_ocr_results(image, results)
save_image(image, "examples/detections.png")
