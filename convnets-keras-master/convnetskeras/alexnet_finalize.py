#!/usr/bin/env python

import numpy as np
import h5py
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard,CSVLogger,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from convnets import convnet
import itertools
# from PIL import Image
from cv2 import imread, resize
import matplotlib.pyplot as plt
from glob import glob
import time
start_time = time.time()

# nohup python train.py &
# ps -ef | grep train.py
# kill UID

same_size = True
model_num = 1
logs_path = "/home/lkara/deepdrive/deepdriving/convnets-keras-master/convnetskeras/models/run%d/" % model_num
model_filename = '/home/lkara/deepdrive/deepdriving/convnets-keras-master/convnetskeras/models/model%d.json' % model_num
weights_filename = '/home/lkara/deepdrive/deepdriving/convnets-keras-master/convnetskeras/models/model%d.h5' % model_num
csvlog_filename = '/home/lkara/deepdrive/deepdriving/convnets-keras-master/convnetskeras/models/model%d.csv' % model_num

##  tensorboard --logdir /home/lkara/deepdrive/deepdriving/convnets-keras-master/convnetskeras/models/
tbCallBack = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
csvlog = CSVLogger(csvlog_filename, separator=',', append=False )
mdlchkpt = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=True, period=2)
erlystp = EarlyStopping(monitor='val_mean_absolute_error',min_delta=1e-4,patience=10) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)


def train(db, keys, avg):
    m = 100000 # len(keys)

    batch_size = 16  # powers of 2
    stream_size = batch_size * 1000  # 16K images loaded at a time
    epochs = 65
    
    model = get_model()

    for i in range(0, m, stream_size):
        print(i, 'iteration')
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        model.fit(X_batch, Y_batch, 
                  batch_size=batch_size, epochs=epochs, 
                  validation_split=0.2, verbose=2, 
                  callbacks=[tbCallBack,csvlog,reduce_lr,mdlchkpt])

    return model


def get_model():
    K.set_image_dim_ordering('th')
    
    base_model = convnet('alexnet', weights_path = 'alexnet_weights.h5')
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

    adam = Adam(lr = 1e-4)
    model.compile(optimizer=adam, loss='mse')
    #sgd = SGD(lr=0.05, decay=0.0005, momentum=0.9)    
    #model.compile(optimizer=sgd, loss='mse')
    return model


def get_data(db, keys, avg):
    n = len(keys)
    if K.image_dim_ordering() == 'tf':
        if same_size:
            X_train = np.empty((n, 210, 280, 3))
        else:
            X_train = np.empty((n, 227, 227, 3))
    else:
        if same_size:
            X_train = np.empty((n, 3, 210, 280))
        else:
            X_train = np.empty((n, 3, 227, 227))
            
    Y_train = np.empty((n, 14))

    for i, key in enumerate(keys):
        img = imread(key)
        # img.shape = 210x280x3
        if K.image_dim_ordering() == 'th':
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)
        if not same_size:
            img = resize(img, (227, 227, 3))

        img = img.astype('float32')
        img = img / 255
        img = np.subtract(img, avg)
        X_train[i] = img

        j = int(key[-12:-4])
        affordances = db[j-1]
        if int(affordances[0]) != j:
            raise ValueError('Image and affordance do not match: ' + str(j))
        affordances = affordances[1:]
        affordances = affordances.reshape(1, 14)
        Y_train[i] = affordances

    return X_train, Y_train


def load_average():
    h5f = h5py.File('../../deepdriving_average.h5', 'r')
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
    model = train(db, keys, avg)

    model.save('alexnet.h5')
    print("Time taken is %s seconds " % (time.time() - start_time))
