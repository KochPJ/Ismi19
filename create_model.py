from keras import Model
from keras.layers import *
from keras.applications import vgg16, resnet50
import keras.backend as K
from keras.utils import get_file
from keras.layers.core import SpatialDropout2D, Activation
from keras.layers import Flatten, Dropout
from keras import layers, Sequential


def createResNet50(in_t, printmodel = False):
    model = resnet50.ResNet50(include_top= None, input_tensor=in_t) #weights='imagenet'

    # freeze the layers
    #for layer in model.layers:
    #    layer.trainable = False

    # get the output of the model
    x = model.layers[-2].output

    x = Flatten()(x)
    
    x = layers.Dense(512, activation='relu')(x)

    x = layers.Dense(256, activation='relu')(x)

    x = layers.BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x_out = layers.Dense(2, activation='softmax')(x)

    ResNet_new = Model(input=in_t, output=x_out)

    if(printmodel):
        ResNet_new.summary()

    return ResNet_new


def CreateKaggleModel(printmodel=False):
    kernel_size = (3, 3)
    pool_size = (2, 2)
    first_filters = 32
    second_filters = 64
    third_filters = 128

    dropout_conv = 0.3
    dropout_dense = 0.5

    model = Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(first_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(second_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(second_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(third_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(third_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(dropout_conv))

    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_dense))
    model.add(Dense(2, activation="softmax"))

    if printmodel:
        print(model.summary())

    return model
