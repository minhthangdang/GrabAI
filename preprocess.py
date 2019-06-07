# import config
from config import config
import csv # required for csv reader
import os # required for paths
import random # required for randomization
import cv2 # opencv
from keras.preprocessing.image import img_to_array # convert image to array
import numpy as np # numpy
from sklearn.preprocessing import LabelBinarizer # required for labels binarizing

# this will build data, labels and mlb from disk
def build_data_and_labels():
    # initialize the data and labels
    data = []
    labels = []

    print("[INFO] loading annotations...")

    # read the class names from csv file and saved it to an array
    with open(config.CLASSNAMES_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        class_names = next(csv_reader)

    # read the annotations from csv file
    annos = dict()
    bounding_box = dict()
    with open(config.ANNOS_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # get the relative path to the image file
            file_path = row[0]

            # get file name from file path car_ims/xxx.jpg
            pos = file_path.rfind("/")
            file_name = file_path[pos + 1:]

            # get class name, it will be something like 'McLaren MP4-12C Coupe 2012'
            class_index = int(row[5]) - 1
            class_name = class_names[class_index]

            # remove year from class name, so it will be like 'McLaren MP4-12C Coupe'
            pos = class_name.rfind(" ")
            class_name = class_name[0:pos]

            # get bounding box from the csv file
            x1 = int(row[1])
            y1 = int(row[2])
            x2 = int(row[3])
            y2 = int(row[4])

            # get make and model from class name, so make  = 'McLaren' and model = 'MP4-12C Coupe'
            '''make = ""
            model = ""
            for _make in config.MAKES:
                if class_name.startswith(_make):
                    make = _make
                    model = class_name.replace(_make+" ", "")

            annos[file_name] = [make, model]'''
            annos[file_name] = class_name
            bounding_box[file_path] = [x1, y1, x2, y2]


    print("[INFO] loading images...")
    image_format = config.IMAGES_PATH + os.path.sep + "{}"
    image_paths = [image_format.format(i) for i in (os.listdir(config.IMAGES_PATH))]
    # image_paths = image_paths[0:100] # TODO: remove this
    image_paths = sorted(image_paths)
    random.seed(config.RANDOM_SEED)
    random.shuffle(image_paths) # randomize for better training

    print("[INFO] building data and labels...")
    count = 0
    for image_path in image_paths:
        count += 1
        print("building data and labels number " + str(count) + " " + image_path)
        # build data
        image = cv2.imread(image_path)
        # extract the car object based on provided bounding box
        coords = bounding_box[image_path]
        x1 = coords[0]
        y1 = coords[1]
        x2 = coords[2]
        y2 = coords[3]
        car_image = image[y1:y2, x1:x2, :]

        # vgg-specific setup
        car_image = cv2.resize(car_image, (config.IMAGE_DIMS[1], config.IMAGE_DIMS[0]))
        car_image = car_image - config.VGG_MEAN # mean subtraction for VGG

        # add car image to data
        car_image = img_to_array(car_image)
        data.append(car_image)

        # build labels
        image_name = os.path.basename(image_path)
        label = annos[image_name] # label contains make and model
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0 # normalize to [0,1] for faster training
    labels = np.array(labels)

    # binarize labels into 1-hot encode vector
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)

    return data, labels, label_binarizer
