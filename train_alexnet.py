#!/usr/bin/env python

import numpy as np
import h5py
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from convnetskeras.convnets import AlexNet
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
weights_filename = '/home/lkara/deepdrive/deepdriving/models/model%d.h5' % model_num
csvlog_filename = '/home/lkara/deepdrive/deepdriving/models/model%d.csv' % model_num

#  tensorboard --logdir /home/lkara/deepdrive/deepdriving/models/
tbCallBack = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)
csvlog = CSVLogger(csvlog_filename, separator=',', append=False)
mdlchkpt = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=True, period=2)
erlystp = EarlyStopping(monitor='val_mean_absolute_error', min_delta=1e-4, patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)

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
    epochs = 65

    model = get_model()

    for i in range(0, m, stream_size):
        print(i, 'iteration')
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        model.fit(X_batch, Y_batch,
                  batch_size=batch_size, epochs=epochs,
                  validation_split=0.2, verbose=2,
                  callbacks=[csvlog, reduce_lr, mdlchkpt, tbCallBack])  # tbCallBack

    return model


def get_model():
    base_model = AlexNet(weights_path='alexnet_weights.h5', dim=dim)
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
        affordances = affordances.reshape(1, 14)
        Y_train[i] = affordances

    return X_train, Y_train


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
