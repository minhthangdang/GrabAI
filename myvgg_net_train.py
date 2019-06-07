# USAGE
# python myvgg_net_train.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import config
from config import config

# import file for preprocessing
import preprocess

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from myvggnet.myvggnet import MyVGGNet
import matplotlib.pyplot as plt
import numpy as np
import pickle

# build data and labels
data, labels, label_binarizer = preprocess.build_data_and_labels()

# split the data into training and testing (80% and 20% respectively)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=config.RANDOM_SEED)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = MyVGGNet.build(classes=len(label_binarizer.classes_), finalAct="softmax")

# initialize the optimizer
opt = Adam(lr=config.LR, decay=config.LR / config.EPOCHS)

# compile the model using binary cross-entropy
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // config.BATCH_SIZE,
	epochs=config.EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(config.MYVGG_MODEL)

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(config.MYVGG_LABEL, "wb")
f.write(pickle.dumps(label_binarizer))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = config.EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig(config.PLOT_PATH)