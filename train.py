import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from alexnet import AlexNet
import plyvel

def train(dbpath):

    data = get_data(dbpath)

    model = AlexNet()
    model.fit(X_train, Y_train,
          batch_size=64, nb_epoch=4700, verbose=1,
          validation_data=(X_test, Y_test))
    # max_iter = #epochs * (training set/training_batch_size) 

    score = model.evaluate(X_test, Y_test, batch_size=64, verbose=1)
    print('Test score:', score)

    model.save('deepdriving_model.h5')
    model.save_weights('deepdriving_weights.h5')
    with open('deepdriving_model.json', 'w') as f:
        f.write(model.to_json())


def get_data(dbpath):
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

    # resize 3x227x227
    # subtract mean
    # crop = 0, mirror = false
    # shuffle

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

    return {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}

