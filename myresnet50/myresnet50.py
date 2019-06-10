# import the necessary packages
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, Sequential
# import our config
from config import config


class MyResNet50:
    @staticmethod
    def build(classes, finalAct="softmax"):
        # initialize ResNet50 network without the fully connected layer
        conv_base = ResNet50(weights="imagenet", include_top=False, input_shape=config.IMAGE_DIMS)
        for layer in conv_base.layers:
            layer.trainable = False

        # setting up our model
        model = Sequential()
        model.add(conv_base)

        model.add(Dense(classes, activation=finalAct))

        return model

