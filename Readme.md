# Photo OCR

Read text in photos / images with complex backgrounds with this easy-to-use Python library.

![Image showing recognized text](images/pub.jpg "Image showing recognized text")

```python
from PIL import Image
from photo_ocr import ocr, draw_ocr_results

image = Image.open("images/pub.jpg")

# run the ocr
text = ocr(image)
print(text)

# draw bounding polygons and text on the image
image = draw_ocr_results(image, text)

# done!
image.save("images/pub_annotated.jpg")
```



### How is this different to tesseract / other OCR tools?

Classic OCR tools like [tesseract](https://github.com/tesseract-ocr/tesseract) work best with scans of book pages / documents. A lot of
manual image pre-processing is needed to get useful results out of these tools
when running them on images that do not contain black-on-white text. In
this case it is best to use tools that specialise on so-called "Scene
text recognition" (e.g. photo_ocr).


|Classic OCR (e.g. tesseract)| Scene text recognition (e.g. photo_ocr) |
--- | --- |
|![Photo of a book page](images/book.jpg "Photo of a book page")|![Photo of a board with many stickers and graffiti](images/stickers.jpg "Photo of a board with many stickers and graffiti")|
|Black text on white background, standard font, layout matters (lines / paragraphs / columns). | Large variation in fonts, font sizes, colours, angles. Mostly individual words, no consistent layout.|


### How does it work?

photo_ocr processes an image in three stages:

| <div style="width:200px">1. Text detection </div> | <div style="width:200px">2. Cropping</div> | <div style="width:200px">3. Text recognition </div>| 
:---: | :---: | :---: |
![](images/hairdresser.jpg) | ![](images/crop0.png) <br/>![](images/crop1.png)  <br/>![](images/crop2.png)| <br/> erisox (0.08, fail!), <br/> <br/>cheri (0.97),<br/><br/>solstudio (0.94) |
Find bounding polygons around words | Crop and align horizontally | "Reading" the text from the cropped images |


### Which languages are supported?

The underlying deep learning models have been trained on __English__ words, but
work well also for other languages that use a __latin alphabet__ (see xx for known issues). 
Other alphabets are currently not supported.

### Which models are supported?

photo_ocr is a wrapper around deep learning models kindly open-sourced by [Clova.ai](https://clova.ai).

For text detection, photo_ocr uses the CRAFT text detection model ([paper](https://arxiv.org/pdf/1904.01941.pdf), [original source code](https://github.com/clovaai/CRAFT-pytorch)). CRAFT 
has been released under MIT license (see file [LICENSE_detection](LICENSE_detection)). 

For text recognition, photo_ocr uses the models released in the Clova.ai text recognition
model benchmark ([paper](https://arxiv.org/pdf/1904.01906.pdf), [original source code](https://github.com/clovaai/deep-text-recognition-benchmark)).
This collection of models has been released under Apache license (see file
[LICENSE_recognition](LICENSE_recognition)). 


## Installation



```
pip install photo_ocr
```


#### Getting the models

All models are automatically downloaded the first time they are needed. You
don't need to do anything.

todo manual download

## Usage

# How to?

```
git clone https://github.com/krasch/photo_ocr.git
pipenv install
pipenv shell
python example.py
```