#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:33:45 2017

@author: lkara
"""

import numpy as np
import h5py
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout,Input, Convolution2D, MaxPooling2D,ZeroPadding2D,LRN2D, merge, Flatten, Activation
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from keras.utils.layer_utils import convert_all_kernels_in_model
#from convnetskeras.convnets import AlexNet
import cv2
from glob import glob
from time import time
start_time = time()

# nohup python train.py &
# ps -ef | grep train.py
# kill UID


same_size = False
model_num = 1
logs_path = "/home/lkara/deepdrive/deepdriving/models/run%d/" % model_num
model_filename = '/home/lkara/deepdrive/deepdriving/models/model%d.json' % model_num
weights_filename = '/home/lkara/deepdrive/deepdriving/models/model_weights%d.h5' % model_num
csvlog_filename = '/home/lkara/deepdrive/deepdriving/models/model%d.csv' % model_num

#  tensorboard --logdir /home/lkara/deepdrive/deepdriving/models/
# tbCallBack = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)
csvlog = CSVLogger(csvlog_filename, separator=',', append=False)
mdlchkpt = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=True, period=2, verbose=1)
erlystp = EarlyStopping(monitor='val_mean_absolute_error', min_delta=1e-4, patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5, verbose=1)

K.set_image_dim_ordering('th')
if K.image_dim_ordering() == 'tf':
    if same_size:
        dim = (210, 280, 3)
    else:
        dim = (227, 227, 3)
else:
    if same_size:
        dim = (3, 210, 280)
    else:
        dim = (3, 227, 227)


def train(db, keys, avg):
    m = 100000  # len(keys)

    batch_size = 16  # powers of 2
    stream_size = batch_size * 500  # 16K images loaded at a time
    epochs = 5
    model = get_model()

    for j in range(epochs):
        for i in range(0, m, stream_size):
            print(i, 'iteration')
            X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
            model.fit(X_batch, Y_batch,
                      batch_size=batch_size, epochs=1,
                      validation_split=0.2, verbose=2,
                      callbacks=[csvlog, reduce_lr, mdlchkpt])  # tbCallBack

    return model


def get_model():
    base_model = AlexNet(weights_path='alexnet_weights.h5', dim=dim)

    # remove softmax output layer

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Dense(512, activation='relu', init='glorot_normal', name='fc1')(x)
    x = Dense(512, activation='relu', init='glorot_normal', name='fc2')(x)
    #x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', init='glorot_normal', name='fc3')(x)
    x = Dense(14, activation='linear', init='glorot_normal', name='out')(x)

    model = Model(input=base_model.input, output=x)
    model.summary()

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='mse')
    # sgd = SGD(lr=0.05, decay=0.0005, momentum=0.9)
    # model.compile(optimizer=sgd, loss='mse')
    return model


def AlexNet(weights_path=None, heatmap=False, dim=(3,227,227)):
    K.set_image_dim_ordering('th')
    inputs = Input(shape=dim)

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    #conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = LRN2D(alpha=1e-4, beta=0.75, n=5)(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")
    
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    #conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = LRN2D(alpha=1e-4, beta=0.75, n=5)(conv_3)  # local response normalization (not batch)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000,name='dense_3')(dense_3)
    prediction = Activation("softmax", name = "softmax")(dense_3)


    model = Model(input=inputs, output=prediction)

    if weights_path:
       model.load_weights(weights_path)

    if K.backend() == 'tensorflow':
        model = convert_all_kernels_in_model(model)

    return model


def get_data(db, keys, avg):
    n = len(keys)

    xdim = (n,) + dim
    X_train = np.empty(xdim)
    Y_train = np.empty((n, 14))

    for i, key in enumerate(keys):
        img = cv2.imread(key)
        # img.shape = 210x280x3
        if not same_size:
            img = cv2.resize(img, (227, 227))

        img = img.astype('float32')

        # convnet preprocessing using during training
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # img = img / 255.0
        # img = np.subtract(img, avg)
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

    return X_train, Y_train


def scale_output(affordances):
    affordances[0] = affordances[0] / 1.1 + 0.5  #angle

    affordances[1] = affordances[1] / 5.6249 + 1.34445     #toMarking_L
    affordances[2] = affordances[2] / 6.8752 + 0.39091     #toMarking_M
    affordances[3] = affordances[3] / 5.6249 - 0.34445     #toMarking_R

    affordances[4] = affordances[4] / 95 + 0.12            #dist_L
    affordances[5] = affordances[5] / 95 + 0.12            #dist_R

    affordances[6] = affordances[6] / 6.8752 + 1.48181     #toMarking_LL
    affordances[7] = affordances[7] / 6.25 + 0.98          #toMarking_ML
    affordances[8] = affordances[8] / 6.25 + 0.02          #toMarking_MR
    affordances[9] = affordances[9] / 6.8752 - 0.48181     #toMarking_RR

    affordances[10] = affordances[10] / 95 + 0.12          #dist_LL
    affordances[11] = affordances[11] / 95 + 0.12          #dist_MM
    affordances[12] = affordances[12] / 95 + 0.12          #dist_RR
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
    h5f = h5py.File('deepdriving_average.h5', 'r')
    avg = h5f['average'][:]
    h5f.close()
    return avg


if __name__ == "__main__":
    dbpath = '/home/lkara/deepdrive/train_images/'
    keys = glob(dbpath + '*.jpg')
    keys.sort()
    db = np.load(dbpath + 'affordances.npy')
    db = db.astype('float32')

    avg = load_average()
    # avg.shape = 210x280x3
    if not same_size:
        avg = cv2.resize(avg, (227, 227))

    model = train(db, keys, avg)

    model.save('alexnet%d.h5' % model_num)
    print("Time taken is %s seconds " % (time() - start_time))