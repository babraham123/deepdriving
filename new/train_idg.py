#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:33:45 2017

@author: Bereket Abraham
"""

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Reshape, Permute, Activation, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
import numpy as np
import h5py
import cv2
from glob import glob
from time import time
from os.path import isfile
from matplotlib import pyplot as plt

start_time = time()

# source activate deepenv1
# nohup python train.py &
# ps -ef | grep train.py
# kill UID

same_size = True
pretrained = False
model_num = 7
folder = "/home/lkara/deepdrive/deepdriving/new/"
logs_path = folder + "models"
model_filename = folder + 'models/cnnmodel%d.json' % model_num
weights_filename = folder + 'models/acnnmodel%d_weights.h5' % model_num
csvlog_filename = folder + 'models/model%d.csv' % model_num

#  tensorboard --logdir /home/lkara/deepdrive/deepdriving/models/
# tbCallBack = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)
csvlog = CSVLogger(csvlog_filename, separator=',', append=False)
mdlchkpt = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=True, period=2, verbose=1)
erlystp = EarlyStopping(monitor='val_mean_absolute_error', min_delta=1e-4, patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5, verbose=1)

if K.image_dim_ordering() == 'tf':
    print('Tensorflow')
    if same_size:
        dim = (210, 280, 3)
    else:
        dim = (227, 227, 3)
else:
    print('Theano')
    if same_size:
        dim = (3, 210, 280)
    else:
        dim = (3, 227, 227)


def train(db, keys, avg):
    #m = len(keys)  # len(keys)
    m = 100000

    batch_size = 128#32  # powers of 2
    #stream_size = batch_size * 500  # 8K images loaded at a time
    epochs = 10
    samples = int(m/batch_size) - 1
    if pretrained and isfile(weights_filename):
        model = alexnet(weights_path=weights_filename)
    else:
        model = alexnet()
    # for layer in base_model.layers:
    #     layer.trainable = False
    # x = base_model.output
    # x = Dense(512, activation='relu', init='glorot_normal', name='fc1')(x)
    model.save('retest_alexnet.h5')
    print('Saved model')
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)

    # train_generator = train_datagen.flow_from_directory()

    model.fit_generator(
        our_datagen(db, keys[0:m], avg, batch_size),
        steps_per_epoch = samples,
        epochs = epochs,
        callbacks=[csvlog, reduce_lr, mdlchkpt])


    return model


def alexnet(weights_path=None):
    """
    Returns a keras model for a CNN.
    input data are of the shape (227,227), and the colors in the RGB order (default)

    model: The keras model for this convnet
    output_dict: Dict of feature layers, asked for in output_layers.
    """
    inputs = Input(shape=dim)

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu', name='conv_1')(inputs)
    # initial weights filler? gaussian, std 0.01
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    # in caffe: Local Response Normalization (LRN)

    # alpha = 1e-4, k=2, beta=0.75, n=5,
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = Convolution2D(256, 5, 5, activation="relu", name='conv_2')(conv_2)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Convolution2D(384, 3, 3, activation="relu", name='conv_4')(conv_4)

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Convolution2D(256, 3, 3, activation="relu", name='conv_5')(conv_5)
    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

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
    dense_4 = Dense(14, activation='linear', name='dense_4')(dense_4)
    # dense_4 = Dense(14, activation='linear', name='dense_4')(dense_4)

    model = Model(input=inputs, output=dense_4)
    model.summary()

    
    if weights_path:
        model.load_weights(weights_path)

    # sgd = SGD(lr=0.01, decay=0.0005, momentum=0.9)  # nesterov=True) # LSTM
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='mse',metrics=['mae'])  # try cross-entropy

    return model


def our_datagen(db, keys, avg,batch_size):
    n = len(keys)/batch_size
    n = int(n)
    for index in range(0,n):
        xdim = (batch_size,) + dim
        X_train = np.empty(xdim)
        Y_train = np.empty((batch_size, 14))

        for i, key in enumerate(keys[index:(index+batch_size)]):
            img = cv2.imread(key)
            # img.shape = 210x280x3
            if not same_size:
                img = cv2.resize(img, (227, 227))

            img = img.astype('float32')

            # convnet preprocessing using during training
            # img[:, :, 0] -= 123.68
            # img[:, :, 1] -= 116.779
            # img[:, :, 2] -= 103.939
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.imshow(img)
            plt.show()
            img = img / 255.0
            img = np.subtract(img, avg)
            if K.image_dim_ordering() == 'th':
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 0, 1)

            X_train[i] = img

            j = int(key[-12:-4])
            affordances = db[j - 1]
            if int(affordances[0]) != j:
                raise ValueError('Image and affordance do not match: ' + str(j))
            affordances = affordances[1:]

            affordances = scale_output(affordances)
            affordances = affordances.reshape(1, 14)
            Y_train[i] = affordances

        yield X_train, Y_train


def scale_output(affordances):
    ''' Scale output between [0.1, 0.9]
    '''
    affordances[0] = affordances[0] / 1.1 + 0.5  # angle

    affordances[1] = affordances[1] / 5.6249 + 1.34445  # toMarking_L
    affordances[2] = affordances[2] / 6.8752 + 0.39091  # toMarking_M
    affordances[3] = affordances[3] / 5.6249 - 0.34445     #toMarking_R

    affordances[4] = affordances[4] / 95 + 0.12     #dist_L
    affordances[5] = affordances[5] / 95 + 0.12     #dist_R

    affordances[6] = affordances[6] / 6.8752 + 1.48181       #toMarking_LL
    affordances[7] = affordances[7] / 6.25 + 0.98             #toMarking_ML
    affordances[8] = affordances[8] / 6.25 + 0.02             #toMarking_MR
    affordances[9] = affordances[9] / 6.8752 - 0.48181        #toMarking_RR

    affordances[10] = affordances[10] / 95 + 0.12  # dist_LL
    affordances[11] = affordances[11] / 95 + 0.12  # dist_MM
    affordances[12] = affordances[12] / 95 + 0.12  # dist_RR
    return affordances


def descale_output(affordances):
    affordances[0] = (affordances[0] - 0.5) * 1.1

    affordances[1] = (affordances[1] - 1.34445) * 5.6249
    affordances[2] = (affordances[2] - 0.39091) * 6.8752
    affordances[3] = (affordances[3] + 0.34445) * 5.6249

    affordances[4] = (affordances[4] - 0.12) * 95
    affordances[5] = (affordances[5] - 0.12) * 95

    affordances[6] = (affordances[6] - 1.48181) * 6.8752
    affordances[7] = (affordances[7] - 0.98) * 6.25
    affordances[8] = (affordances[8] - 0.02) * 6.25
    affordances[9] = (affordances[9] + 0.48181) * 6.8752

    affordances[10] = (affordances[10] - 0.12) * 95
    affordances[11] = (affordances[11] - 0.12) * 95
    affordances[12] = (affordances[12] - 0.12) * 95
    return affordances


def load_average():
    h5f = h5py.File('/home/lkara/deepdrive/deepdriving/deepdriving_average.h5', 'r')
    avg = h5f['average'][:]
    h5f.close()
    return avg


if __name__ == "__main__":
    dbpath = '/home/lkara/deepdrive/train_images/'
    keys = glob(dbpath + '*.jpg')
    keys.sort()
    db = np.load(dbpath + 'affordances.npy')

    # TODO : shuffle and keep aligned

    db = db.astype('float32')

    avg = load_average()
    # avg.shape = 210x280x3
    if not same_size:
        avg = cv2.resize(avg, (227, 227))

    model = train(db, keys, avg)

    model.save(folder + "models/acnn%d.h5" % model_num)
    print("Time taken is %s seconds " % (time() - start_time))
