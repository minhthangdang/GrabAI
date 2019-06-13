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
learning model - myvgg.model](https://drive.google.com/open?id=14tsq_x5b4CP8gzaFh_It3ZqzV8owkoC6) 
and put it under **"models"** folder in my codebase.

In my classifying process, I've utilised [YOLO from Darknet](https://pjreddie.com/darknet/yolo/) for the task of car localisation. 
So in addition to my deep learning model above, you also need to download 
[the weights for YOLO - yolov3.weights](https://drive.google.com/open?id=1PAba0klLoELLp9F1DaGAwGyVXXQGOCA0)
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

| Ford Fiesta | Toyota Camry Sedan |
| --- | --- |
| ![Ford Fiesta Sedan](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Ford-Fiesta-Sedan.JPG) | ![Toyota Camry Sedan](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Toyota-Camry-Sedan.JPG) |

| Hyundai Santa Fe SUV | Cadillac CTS-V Sedan |
| --- | --- |
| ![Hyundai Santa Fe SUV](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Hyundai_SantaFe_SUV.JPG) | ![Cadillac CTS-V Sedan](https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/Cadillac-CTSV-Sedan.JPG) | 


### Training

A trained model has been provided in the link above, which is ready for classification usage. However if by any chance you would like to re-run
the model training, please follow the below steps.

First you need to download the Cars dataset provided by Stanford [here](http://imagenet.stanford.edu/internal/car196/car_ims.tgz).
After it's downloaded, put it to the root folder of this repository and unpack it by running:

```
tar -xvzf car_ims.tgz
```

It will unpack all the images to "car_ims" folder. We can now run the script which trains the model:

```
python myvgg_net_train.py
```

Please note that you will probably need a GPU to train my model. For your reference, I run my model training on Amazon 
Web Service with a Tesla K80 GPU (the p2.xlarge EC2 instance).

It may take a good few hours depending on your machine specification. After it completes, it will create a pickle file named myvgg_labels.pickle 
and four models named model_1.model, model_2.model, model_3.model and myvgg.model in the "models" folder. The first three models are created from
training three fine-tuned VGGNet, and the last model is an ensemble model of the first three. I will explain the technical details
of the training process later. For now you can use the myvgg.model file for classification as described in the section above.

## Technical Details

### Preparation

The Stanford Cars dataset is accompanied with a [devkit](https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz) which includes
the class labels and bounding boxes for all images. These are provided in matlab format and I've converted them into
CSV files for easy manipulation. These CSV files are located under "annotations" folder where:

* *annotations.csv* contains the annotations (image path, label id, bounding box coordinates, etc.) for each image.

* *class_names.csv* contains all the class names.

### Preprocess

A preprocess step is carried out before the actual model training. This is written in the method *build_data_and_labels* in the file *preprocess.py*

In this method, the following are performed:

* Read the annotations and remove the year from the class names, so that it's left with make and model only

* Utilise the bounding box and retain the image section within the bounding box only. We will use these cropped images
for training rather than the whole images as it reduces the noise and yields better performance.

* In my model training, I use VGG16 network as the base model, so each image is normalised according to VGG-specific setup 
such as image resize (224x224), mean subtraction,  etc.


### Feature Engineering

Since 2013 when the Stanford Cars dataset was first introduced, hand-crafted features for classification problems have been
fade away and replaced by deep learning models. It has been proved that deep learning models have performed
extremely well such as in the ImageNet Large Scale Visual Recognition Challenge where GoogLeNet and VGGNet was the winner 
and runner-up respectively in 2014, and ResNet was the winner in 2015. At the moment deep learning is the de-facto choice
for image classification as well as many other computer vision problems.

In my project I applied the transfer learning method where I re-used an existing pre-trained deep learning network as the starting point for 
the task of predicting make and model of a car image. I decided to pick out VGG16 and ResNet50 for experiments.

After playing around with both models, it appeared that the VGG16 performed better ResNet50 for the sake of this task. Therefore
I discarded the ResNet50 model and continued with VGG16.


 

## Room for Improvement

* As with many other deep learning models, the more data the better. The dataset provided by Stanford has 16,185 images
for 196 make and model classes. This is considered "small" data in the world of deep learning. Moreover there are imbalances 
where some classes are overrepresented (e.g. Audi and BMW) while others (e.g. Tesla) have only a few dozens. Hence one way
to improve the performance of the system is to feed more data to the neural networks. This will definitely increase the
accuracy of the prediction. Within the time constraint of this challenge, I did not have enough time to collect more data, but
this is a key point for future enhancement.

* In my project I combined three deep learning models into an ensemble, and each model was run with 10 epochs. This could be
improved by having more deep learning models (5 to 10 models) for the ensemble and more epochs for each model. Again this
was not done due to time and resources constraint.

* Experiments with more networks. During the course of this project, I experimented with only 2 deep learning networks: 
VGG16 and ResNet50. There are several other networks such as AlexNet, GoogLeNet, etc. which I did not have time to try.
Better performance may be yielded by exploring other networks.









