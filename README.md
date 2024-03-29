# Photo OCR

Read text in photos / images with complex backgrounds with this easy-to-use Python library. 
Based on deep learning (torchvision) models released by [Clova AI Research ](https://github.com/clovaai/).

![Image showing recognized text](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/pub.jpg "Image showing recognized text")

```python
from PIL import Image
from photo_ocr import ocr, draw_ocr

# (download example.jpg here: https://github.com/krasch/photo_ocr/blob/master/example.jpg)
image = Image.open("example.jpg")

# run the ocr
results = ocr(image)
print(results)

# draw bounding polygons and text on the image
image = draw_ocr(image, results)

# done!
image.save("example_ocr.jpg")
```

## Table of contents

1. [Introduction](#section-introduction)
2. [Installation](#section-installation)
3. [Usage](#section-usage)
4. [Troubleshooting](#section-troubleshooting)
5. [Detailed configuration / parameters](#section-parameters)
4. [Licensing](#section-licensing)

## <a id="section-introduction">1. Introduction</a>


### How is this different to tesseract / other OCR tools?

Classic OCR tools like [tesseract](https://github.com/tesseract-ocr/tesseract) work best with scans of book pages / documents. A lot of
manual image pre-processing is needed to get useful results out of these tools
when running them on images that do not contain black-on-white text. In
this case it is best to use tools that specialise on so-called "Scene
text recognition" (e.g. photo_ocr).


|Classic OCR (e.g. tesseract)| Scene text recognition (e.g. photo_ocr) |
--- | --- |
|![Photo of a book page](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/book.jpg "Photo of a book page")|![Photo of a board with many stickers and graffiti](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/stickers.jpg "Photo of a board with many stickers and graffiti")|
|Black text on white background, standard font, layout matters (lines / paragraphs / columns). | Large variation in fonts, font sizes, colours, angles. Mostly individual words, no consistent layout.|


### How does it work?

photo_ocr processes an image in three stages:

| <div style="width:200px">1. Text detection </div> | <div style="width:200px">2. Cropping</div> | <div style="width:200px">3. Text recognition </div>| 
:---: | :---: | :---: |
![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/hairdresser.jpg) | ![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/crop0.png) <br/>![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/crop1.png)  <br/>![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/crop2.png)| <br/> erisox (0.08, fail!), <br/> <br/>cheri (0.97),<br/><br/>solstudio (0.94) |
Find bounding polygons around words | Crop and align horizontally | "Reading" the text from the cropped images |

### Which models are supported?

photo_ocr is a wrapper around deep learning models kindly open-sourced by [Clova AI Research](https://github.com/clovaai/).

For text detection, photo_ocr uses the CRAFT text detection model ([paper](https://arxiv.org/pdf/1904.01941.pdf), [original source code](https://github.com/clovaai/CRAFT-pytorch)). CRAFT 
has been released under MIT license (see file [LICENSE_detection](LICENSE_detection)). 

For text recognition, photo_ocr uses the models released in the Clova.ai text recognition
model benchmark ([paper](https://arxiv.org/pdf/1904.01906.pdf), [original source code](https://github.com/clovaai/deep-text-recognition-benchmark)).
This collection of models has been released under Apache license (see file
[LICENSE_recognition](LICENSE_recognition)). 


### Which languages are supported?

The models have been trained on __English__ words, but
work well also for other languages that use a __latin alphabet__ (see [Troubleshooting](#section-troubleshooting) for known issues). 
Other alphabets are currently not supported by photo_ocr.


## <a id="section-installation">2. Installation</a>

Works with __python 3.6, 3.7 and 3.8__ 

photo_ocr works with torchvision >=0.7 and <=0.10. If there is a newer version of torchvision which is not yet supported
by photo_ocr, please open a github issue to let us know!

```
git clone https://github.com/krasch/photo_ocr.git
python setup.py install

# check that everything is working
python example.py
```



All __models are automatically downloaded__ the first time they are needed. The
models are stored locally in the standard pytorch model directory, which
you can change by setting the `TORCH_HOME` environment variable (see the [official pytorch documentation](https://pytorch.org/docs/stable/hub.html) for details).


## <a id="section-usage">3. Usage</a>

(You can find a script containing all the snippets below at [example.py](https://github.com/krasch/photo_ocr/blob/master/example.py))

### Input

The library takes as input a [pillow / PIL](https://pillow.readthedocs.io/en/stable/) image.

You can use PIL directly to read the image from file.

```python
from PIL import Image

# (download example.jpg here: https://github.com/krasch/photo_ocr/blob/master/example.jpg)
image = Image.open("example.jpg")
```
For convenience, photo_ocr also offers a `load_image` function, which
opens the image and rotates it according to the EXIF metadata, if necessary.

```python
from photo_ocr import load_image

image = load_image("example.jpg")
```
### Running the OCR

Just one simple function call to the `ocr` function:

```python
from photo_ocr import ocr

results = ocr(image)
```

The `ocr` function returns a list of all text instances found
in the image. The list is sorted by recognition confidence, 
starting with the most confident recognition. 

You can loop over the results like this:

```python
for result in results:
    # polygon around the text
    # (list of xy coordinates: [(x0, y0), (x1, y1),  ....])
    print(result.polygon)
    
    # the actual text (a string)
    print(result.text)
    
    # the recognition confidence (a number in [0.0, 1.0])
    print(result.confidence)
```

Since each entry in the results list is a`namedtuple`, you can
also loop over the results like this: 

```python
for polygon, text, confidence in results:
    print(polygon)
    print(text)
    print(confidence)
```

### Visualising the results

Use the `draw_ocr` method to draw the OCR results onto the original image.

```python
from photo_ocr import draw_ocr

image = draw_ocr(image, results)
image.save("example_ocr.jpg")
```

### Running only text detection

Use the `detection` function to only run the text detection step:

```python
from photo_ocr import detection

# list of polygons where text was found
polygons = detection(image)

for polygon in polygons:
    # polygon around the text
    # (list of xy coordinates: [(x0, y0), (x1, y1),  ....])
    print(polygon)
```

You can use the `draw_detections` function to draw the
results of the `detection`:

```python
from photo_ocr import draw_detections

image = draw_detections(image, polygons)
image.save("example_detections.jpg")
```

### Running only text recognition

Use the `recognition` function to only run the text recognition step.
You need to supply an image that has already been cropped to a
text polygon. The text should be aligned horizontally. 

```python
from photo_ocr import load_image, recognition

# (download crop.jpg here: https://github.com/krasch/photo_ocr/blob/master/crop.jpg)
crop = load_image("crop.jpg")

text, confidence = recognition(crop)
```

### GPU usage

If you have a GPU, photo_ocr will automatically use it! 

If you have multiple GPUs and want photo_ocr to use a different
one, you can set the `CUDA_VISIBLE_DEVICES` environment
variable, as shown below. Make sure that you import photo_ocr only after you
have set the environment variable!

```python
import os

# if you have e.g. 4 GPUs, you can set their usage order like this
# (photo_ocr will only look at the first entry in the list 
#  and ignore the others, since it runs only one GPU)
os.environ["CUDA_VISIBLE_DEVICES"]="1,0,2,3"

# you can also choose to run on cpu despite having a GPU
# (=simply make no device visible to photo_ocr)
os.environ["CUDA_VISIBLE_DEVICES"]=""

# only import photo_ocr after you have set the environment variable
# otherwise photo_ocr will use the wrong GPU!
from photo_ocr import ocr
```

## <a id="section-troubleshooting">4. Troubleshooting</a>

### Troubleshooting OCR results

| Example | Description |  Reason | Solution | 
:---: | :--- | :--- | :--- |
![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/umlaut.jpg) | Special letters (e.g. å, ö, ñ) are not recognized properly | The models have been trained on latin letters only. In most cases, the recognition still works well, with the model using similar-looking substitutes for the special letters. | Use a spellchecker after running text recognition to get the correct letters. |
![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/gol.jpg) | Special characters (e.g. !, ?, ;) are not recognized properly | The default text recognition model supports only the characters a-z and 0-9. | Switch to the <a href="#param-model">case-sensitive model</a>, which also supports 30 common special characters. 
![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/angle.jpg)  | Text area is found, but text recognition returns only one-letter results (e.g. e, i, a) | The angle of the text is so steep, that the crop is being rotated in the wrong direction. | Rotate the input image by 90°. |
![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/borders.jpg)  | Text area is not found. | - | Try decreasing the <a href="#param-confidence_threshold">confidence threshold</a>. Alternatively, decrease the <a href="#param-text_threshold_first_pass">text_threshold_first_pass</a> and <a href="#param-text_threshold_second_pass">text_threshold_second_pass</a>. |
![](https://raw.githubusercontent.com/krasch/photo_ocr/master/docs/images/cow.jpg)  | Text area is found where there is no text. | - | Try increasing the  <a href="#param-confidence_threshold">confidence threshold</a>. Alternatively, increase the <a href="#param-text_threshold_first_pass">text_threshold_first_pass</a> and <a href="#param-text_threshold_second_pass">text_threshold_second_pass</a>.  |

### Speeding things up

If photo_ocr is too slow for your use case, try first to identify
if the detection or the recognition step (or both) are running
slow on your images by running both steps in isolation (see above).

To speed up __detection__, try decreasing the <a href="#param-image_max_size">image_max_size</a> 
and/or the <a href="#param-image_magnification">image_magnification</a>. This will result in smaller
images being fed to the detection model, resulting in a faster text detection. An unwanted side effect
might be that smaller text areas are no longer found by the model in the smaller image.

To speed up __recognition__, try switching to a faster <a href="#param-model">model</a>. You might lose some
recognition confidence, though. You can also try increasing the <a href="#param-batch_size">batch_size</a>  (this
makes most sense if your images contain a lot of text instances).


## <a id="section-parameters">5. Detailed configuration / parameters  </a>


### Runtime parameters

| Name | Description |  Values | 
:--- | :--- | :--- 
<a id="param-confidence_threshold">confidence_threshold</a> | Only recognitions with confidence larger than this threshold will be returned. | a float in [0.0, 1.0), default=0.3


```python

results = ocr(image, confidence_threshold=0.3)
```


### Initialisation parameters

For convenience, the `ocr`, `detection`, `recognition` methods
are pre-initialised with sensible defaults. If you want to change any of these parameters, you 
need to initialise these methods again with your own settings (see [initialisation code](#section-initialisation-code)).

##### Detection params

| Name | Description | Values |
:--- | :--- | :--- 
<a id="param-image_max_size">image_max_size</a> | During image pre-processing before running text detection, the image will be resized such that the larger side of the image is smaller than image_max_size. | an integer, default=1280
<a id="param-image_magnification">image_magnification</a> | During image pre-processing before running text detection, the image will be magnified by this value (but no bigger than image_max_size) | a float &ge; 1.0, default=1.5
<a id="param-combine_words_to_lines">combine_words_to_lines</a> | If true, use the additional "RefineNet" to link individual words that are near each other horizontally together.| a boolean, default=False|
<a id="param-text_threshold_first_pass">text_threshold_first_pass</a> | The CRAFT model produces for every pixel a score of howlikely it is that this pixel is part of a text character (called regions score in the paper). During postprocessing, only those pixels are considered, that are above the text_threshold_first_pass. | a float in [0.0, 1.0], default=0.4 | 
<a id="param-text_threshold_second_pass">text_threshold_second_pass</a> | See explanation of text_threshold_first_pass. During postprocessing, there is a second round of thresholding happening after the individual characters have been linked together to words (see link_threshold); detection_text_threshold_second_pass <= detection_text_threshold_first_pass| a float in [0.0, 1.0], default=0.7|
<a id="param-link_threshold">link_threshold</a> | The CRAFT model produces for every pixels a score of how likely it is that this pixel is between two text characters (called affinity score in the paper). During postprocessing, this score is used to link individual characters together as words.| a float in [0.0, 1.0], default=0.4|

##### Recognition params

| Name | Description | Values |
:--- | :--- | :--- 
<a id="param-model">model</a> | Which recognition model to use, see the [paper](https://arxiv.org/pdf/1904.01906.pdf), in particular Figure 4. <br/><br/> Best performance: TPS_ResNet_BiLSTM_Attn <br/><br/>slightly worse performance but five times faster: model_zoo.None_ResNet_None_CTC <br/><br/>case-sensitive: model_zoo.TPS_ResNet_BiLSTM_Attn_case_sensitive| One of the initialisation functions in the photo_ocr.recognition.model_zoo, default=model_zoo.TPS_ResNet_BiLSTM_Attn |
<a id="param-image_width">image_width</a> | During image pre-processing, the (cropped) image will be resized to this width models were trained with width=100, other values don't seem to work as well | an integer, default=100|
<a id="param-image_height">image_height</a> | During image pre-processing, the (cropped) image will be resized to this height;  models were trained with height=32, other values don't seem to work as well | an integer, default=32|
<a id="param-keep_ratio">keep_ratio</a> | When resizing images during pre-processing: True -> keep the width/height ratio (and pad appropriately) or False -> simple resize without keeping ratio| a boolean, default=False| 
<a id="param-batch_size">batch_size</a> | Size of the batches to be fed to the model. | an integer, default=32| 


##### <a id="section-initialisation-code">Initialisation code<a>

```python

from photo_ocr import PhotoOCR
from photo_ocr.recognition import model_zoo 


detection_params = {"image_max_size": 1280,
                    "image_magnification": 1.5,
                    "combine_words_to_lines": False,
                    "text_threshold_first_pass": 0.4,
                    "text_threshold_second_pass": 0.7,
                    "link_threshold": 0.4}

recognition_params = {"model": model_zoo.TPS_ResNet_BiLSTM_Attn,
                      "image_width": 100,
                      "image_height": 32,
                      "keep_ratio": False}

# initialise the photo_ocr object
photo_ocr = PhotoOCR(detection_params, recognition_params)

# optionally: make class methods available as global functions for convenience
ocr = photo_ocr.ocr
detection = photo_ocr.detection
recognition = photo_ocr.recognition
```


## <a id="section-licensing">Licensing</a>

This repository contains three license files:

| Filename | License | Owner | What does it cover? |
:--- | :--- | :---  | :--- |
| LICENSE_detection.txt (Copy of [original license](https://github.com/clovaai/CRAFT-pytorch/blob/6c809d409996d4516e393f04e89965f070ecc14a/LICENSE))| MIT | NAVER Corp. | The model architectures in photo_ocr.detection as well as some of the postprocessing code. Also the detection model weights hosted on [https://github.com/krasch/photo_ocr_models](https://github.com/krasch/photo_ocr_models/) |
| LICENSE_recognition.txt  (Copy of [original license](https://github.com/clovaai/deep-text-recognition-benchmark/blob/2b38eb25749519c9a3646e19628cd7b6a398043d/LICENSE.md))| Apache 2.0 | original license file does not contain a copyright owner, but presumably also NAVER Corp.| The model architectures in photo_ocr.recognition as well as some of the postprocessing code. Also the recognition model weights hosted on [https://github.com/krasch/photo_ocr_models](https://github.com/krasch/photo_ocr_models/) |
| LICENSE.txt | Apache 2.0 | krasch | Everything else |
