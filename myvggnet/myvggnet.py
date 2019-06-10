# import the necessary packages
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.applications import VGG16
from keras import layers
# import our config
from config import config


class MyVGGNet:
    @staticmethod
    def build(classes, finalAct="softmax"):
        # initialize VGG16 network without the fully connected layer
        conv_base = VGG16(weights="imagenet", include_top=False, input_shape=config.IMAGE_DIMS)
        for layer in conv_base.layers:
            layer.trainable = False

        # setting up our model
        model = Sequential()
        model.add(conv_base)

        # fully connected layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # classifier
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        return model

    @staticmethod
    def ensemble_models(models, model_input):
        # collect outputs of models in a list
        y_models = [model(model_input) for model in models]
        # averaging outputs
        y_avg = layers.average(y_models)
        # build model from same input and avg output
        model_ens = Model(inputs=model_input, outputs=y_avg, name='ensemble')

        return model_ens
