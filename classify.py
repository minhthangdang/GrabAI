# USAGE
# python classify.py --image examples/example_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
from config import config
from car_detect import detect_bounding_box
import utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
original = image.copy()

# get the car bounding box
(x, y, width, height) = detect_bounding_box(image)

# extract the area which contains the car
image = image[y:y + height, x:x + width]

# pre-process the image for classification
image = cv2.resize(image, (config.IMAGE_DIMS[1], config.IMAGE_DIMS[0]))
image = image - config.VGG_MEAN
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the label binarizer
print("[INFO] loading vgg network...")
model = load_model(config.MYVGG_MODEL)
label_binarizer = pickle.loads(open(config.MYVGG_LABEL, "rb").read())

# classify the input image then find the index of the class with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argsort(proba)[-1]

# build the label and draw the label on the image
original, ratio = utils.resize(original, width = 800)
label = "{}: {:.2f}%".format(label_binarizer.classes_[idx], proba[idx] * 100)
cv2.putText(original, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# draw the bounding box
x = int(x*ratio)
y = int(y*ratio)
width = int(width*ratio)
height = int(height*ratio)
cv2.rectangle(original, (x, y), (x+width, y+height), (0, 255, 0), 2)

# show the output image with bounding box
cv2.imshow("Output", original)
cv2.waitKey(0)