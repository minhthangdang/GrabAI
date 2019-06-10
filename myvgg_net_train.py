# USAGE
# python myvgg_net_train.py

# import config
from config import config

# import file for preprocessing
import preprocess

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from myvggnet.myvggnet import MyVGGNet
import pickle
import utils

# build data and labels
data, labels, label_binarizer = preprocess.build_data_and_labels()

# split the data into training and testing (80% and 20% respectively)
print("[INFO] splitting data for train/test...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=config.RANDOM_SEED)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = MyVGGNet.build(classes=len(label_binarizer.classes_), finalAct="softmax")

# initialize the optimizer
# opt = Adam(lr=config.LR, decay=config.LR / config.EPOCHS)
# opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)
# opt = RMSprop(lr=0.0001)
opts = []
opts.append(Adam(lr=config.LR, decay=config.LR / config.EPOCHS))
opts.append(RMSprop(lr=0.0001))

for idx, opt in enumerate(opts):

	# compile the model using binary cross-entropy
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	# train the network
	print("[INFO] training network " + str(idx) + "...")
	H = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // config.BATCH_SIZE,
		epochs=config.EPOCHS, verbose=1)

	# save the model to disk
	print("[INFO] serializing network...")
	model.save(config.MYVGG_MODEL + str(idx))

	# save the label binarizer to disk
	print("[INFO] serializing label binarizer...")
	f = open(config.MYVGG_LABEL + str(idx), "wb")
	f.write(pickle.dumps(label_binarizer))
	f.close()

	# print out classification report on test data
	print("[INFO] preparing classification report...")
	predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
	report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_binarizer.classes_)
	print(report)
	print("[INFO] saving classification report...")
	f = open(config.MYVGG_REPORT_PATH + str(idx), "w")
	f.write(report)

	# plot loss and accuracy and save to file
	# print("[INFO] saving loss and accuracy plot...")
	# utils.plot_loss_accuracy(H)
