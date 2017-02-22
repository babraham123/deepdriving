import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from alexnet import AlexNet
import plyvel

def train(dbpath):

    # the data, shuffled and split between tran and test sets
    # load_data(datapath) from leveldb
    db = plyvel.DB(dbpath)
    i = 0
    for key, value in db:
        print(key)
        print(value)
        i += 1
        if(i < 10):
            break

    db.close()

    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)

    model.fit(X_train, Y_train,
          batch_size=128, nb_epoch=4, verbose=1, # show_accuracy=True
          validation_data=(X_test, Y_test))

    # LOOK AT LOSS
    score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

