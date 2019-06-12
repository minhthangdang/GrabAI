# Grab AI's Computer Vision Challenge 2019

## Introduction

This project is created as a submission for the [Grab AI's Computer Vision Challenge 2019](https://www.aiforsea.com/computer-vision)

## Problem Statement

Given a [dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) of distinct car images , can you automatically recognize the car model and make?

## Installation

The code was written and tested on Ubuntu 16.04, and the following packages were used:
* Python 3.6.7
* OpenCV 4.0.0
* Numpy 1.16.2
* scikit-learn 0.20.3
* matplotlib 3.0.2
* Tensorflow 1.12.0
* Keras 2.2.4

Make sure you have these installed in order to run the code successfully.

## Usage

### Classification

First please clone or download this repository.

A deep learning model has been created for classification task. However, this model is too large for github, so I've uploaded it 
to Google drive. You need to download [my deep
learning model here](https://drive.google.com/open?id=14tsq_x5b4CP8gzaFh_It3ZqzV8owkoC6) 
and put it under **"models"** folder in my codebase.

In my classifying process, I've utilised [YOLO from Darknet](https://pjreddie.com/darknet/yolo/) for the task of car localisation. 
So in addition to my deep learning model above, you also need to download 
[the weights for YOLO here](https://drive.google.com/open?id=1PAba0klLoELLp9F1DaGAwGyVXXQGOCA0)
and put it under **"yolo-coco"** folder in my codebase.

Having done the above steps, we are now ready to do the interesting stuff. To perform the classification task, please
run the following script:

```
python classify.py --image samples/Ford_FiestaSedan.jpg
```

In which "samples/Ford_FiestaSedan.jpg" is the path to the image.

If everything is configured correctly you can see the process running:

![Classification running](https://github.com/minhthangdang/minhthangdang.github.io/raw/master/running-classification.JPG)

It will output the make, model and confidence score for the car in the image, for example "Ford Fiesta Sedan: 47.69%". It will also show the
image with a bounding box for the car and the prediction. Here are a few examples:

![Ford Fiesta Sedan](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Ford-Fiesta-Sedan.JPG)

![Ford GT Coupe](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Ford-GT-Coup.JPG)

![Hyundai Santa Fe SUV](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Hyundai_SantaFe_SUV.JPG)

![Jeep Patriot SUV](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Jeep-Patriot-SUV.JPG) 






