import numpy as np


# define images path
IMAGES_PATH = "car_ims"
# define annotations file
ANNOS_PATH = "annotations.csv"
# define class names file
CLASSNAMES_PATH = "class_names.csv"
# define plot path
PLOT_PATH = "plot.png"
# define list of makes
MAKES = ['Acura', 'AM General', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti', 'Buick',
		 'Cadillac', 'Chevrolet', 'Chrysler', 'Daewoo', 'Dodge', 'Eagle', 'Ferrari', 'FIAT',
		 'Fisker', 'Ford', 'Geo', 'GMC', 'Honda', 'HUMMER', 'Hyundai', 'Infiniti', 'Isuzu',
		 'Jaguar', 'Jeep', 'Lamborghini', 'Land Rover', 'Lincoln', 'Maybach', 'Mazda', 'McLaren',
		 'Mercedes-Benz', 'MINI Cooper', 'Mitsubishi', 'Nissan', 'Plymouth', 'Porsche', 'Ram C/V',
		 'Rolls-Royce', 'Scion', 'smart', 'Spyker', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen',
		 'Volvo']
# image dimensions for training
IMAGE_DIMS = (224, 224, 3)
# number of EPOCHS
EPOCHS = 5
# learning rate
LR = 0.001
# batch size
BATCH_SIZE = 32
# random seed
RANDOM_SEED = 42

# VGG Mean as used by tensorflow https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py
VGG_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)

# define files for vgg model and labels
MYVGG_MODEL="cars.model"
MYVGG_LABEL="labels.pickle"

YOLO_WEIGHTS = "yolo-coco/yolov3.weights"
YOLO_CFG = "yolo-coco/yolov3.cfg"
YOLO_NAMES = "yolo-coco/coco.names"
YOLO_MIN_CONFIDENCE = 0.7
YOLO_NMS_THRESHOLD = 0.4
