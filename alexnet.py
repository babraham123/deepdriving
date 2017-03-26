from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave

from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids

    """
    Returns a keras model for a CNN.
    input data are of the shape (227,227), and the colors in the RGB order (default)
 
    model: The keras model for this convnet
    output_dict: Dict of feature layers, asked for in output_layers.
    """

def AlexNet(weights_path=None):
    inputs = Input(shape=(3, 280, 210)) # input size

    conv_1 = Convolution2D(96, 11, 11, subsample=(4,4), activation='relu',
                           name='conv_1')(inputs)
    # initial weights filler? gaussian, std 0.01
    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    # in caffe: Local Response Normalization (LRN)
    # alpha = 1e-4, k=2, beta=0.75, n=5,
    conv_2 = ZeroPadding2D((2,2))(conv_2)

    # conv_2 = merge([
    #     Convolution2D(128, 5, 5, activation="relu", name='conv_2_'+str(i+1))(
    #         splittensor(ratio_split=2, id_split=i)(conv_2)
    #     ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    # split unnecessary on modern GPUs, no stride
    conv_2 = Convolution2D(256, 5, 5, activation="relu", name='conv_2')(conv_2)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    # conv_4 = merge([
    #     Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
    #         splittensor(ratio_split=2,id_split=i)(conv_4)
    #     ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    # split unnecessary on modern GPUs, no stride
    conv_4 = Convolution2D(384, 3, 3, activation="relu", name='conv_4')(conv_4)

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    # conv_5 = merge([
    #     Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
    #         splittensor(ratio_split=2,id_split=i)(conv_5)
    #     ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")
    
    # split unnecessary on modern GPUs, no stride
    conv_5 = Convolution2D(256, 3, 3, activation="relu", name='conv_5')(conv_5)
    dense_1 = MaxPooling2D((3, 3), strides=(2,2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    # initial weights filler? gaussian, std 0.005
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)

    # initial weights filler? gaussian, std 0.01
    dense_3 = Dense(256, activation='relu', name='dense_3')(dense_3)
    dense_4 = Dropout(0.5)(dense_3)

    # output: 14 affordances, gaussian std 0.01
    dense_4 = Dense(14, activation='sigmoid', name='dense_4')(dense_4)

    model = Model(input=inputs, output=dense_4)

    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=0.01, decay=0.0005, momentum=0.9) # nesterov=True)
    # caffe: euclidean loss
    model.compile(optimizer=sgd, loss='mean_squared_error')

    return model
