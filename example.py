
from photo_ocr import load_image, save_image, ocr, draw_ocr_results

# load the image
image = load_image("examples/rifles.png")

# run the ocr
results = ocr(image)
results = sorted(results, key = lambda item: item.confidence, reverse=True)
for r in results:
    print(r.word, r.confidence)

# draw the results on the image and save image
image = draw_ocr_results(image, results)
save_image(image, "examples/detections.png")
