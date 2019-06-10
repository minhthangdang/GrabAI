# import the necessary packages
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
# import our config
from config import config


class MyResNet50:
    @staticmethod
    def build(classes, finalAct="softmax"):
        # initialize ResNet50 network without the fully connected layer
        conv_base = ResNet50(weights="imagenet", include_top=False, input_shape=config.IMAGE_DIMS)
        for layer in conv_base.layers:
            layer.trainable = False

        fc_layers = [1024, 1024]
        x = conv_base.output
        x = Flatten()(x)
        for fc in fc_layers:
            # New FC layer, random init
            x = Dense(fc, activation='relu')(x)
            x = Dropout(0.5)(x)

        # New softmax layer
        predictions = Dense(len(classes), activation=finalAct)(x)

        model = Model(inputs=conv_base.input, outputs=predictions)

        return model

