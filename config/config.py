import numpy as np


# define images path
IMAGES_PATH = "car_ims"
# define annotations file
ANNOS_PATH = "annotations/annotations.csv"
# define class names file
CLASSNAMES_PATH = "annotations/class_names.csv"
# image dimensions for training
IMAGE_DIMS = (224, 224, 3)
# number of EPOCHS
EPOCHS = 5
# learning rate
LR = 0.0001
# batch size
BATCH_SIZE = 32
# random seed
RANDOM_SEED = 42

# define MYVGG model plot path
MYVGG_PLOT_PATH = "plot.png"

# define files for vgg model and labels
MYVGG_MODEL = "myvgg.model"
MYVGG_LABEL = "myvgg_labels.pickle"

YOLO_WEIGHTS = "yolo-coco/yolov3.weights"
YOLO_CFG = "yolo-coco/yolov3.cfg"
YOLO_NAMES = "yolo-coco/coco.names"
YOLO_MIN_CONFIDENCE = 0.7
YOLO_NMS_THRESHOLD = 0.4
