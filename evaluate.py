import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from alexnet import AlexNet
import plyvel

def train(db, keys):
    m = len(keys)
    # 19 epochs
    iterations = 10 # 140000
    batch_size = 64
    stream_size = batch_size * 150 # ~10K images loaded at a time

    model = AlexNet()

    for i in range(0, m, stream_size):
        X_train, Y_train = get_data(db, keys[i:(i+stream_size)])
        model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1, verbose=1)

    # model.fit(X_train, Y_train,
    #       batch_size=64, nb_epoch=4700, verbose=1,
    #       validation_data=(X_test, Y_test))
    # max_iter = #epochs * (training set/training_batch_size) 

    score = model.evaluate(X_test, Y_test, batch_size=64, verbose=1)
    print('Test score:', score)

    return model


def get_data(dbpath, keys):
    for key in keys:
        datum = caffe_pb2.Datum.FromString(db.get(key))
        img = caffe.io.datum_to_array(datum)
        # img.shape = 3x210x280
        affordances = [i for i in datum.float_data]

    # resize 3x210x280
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


if __name__ == "__main__":
    dbpath = '../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/'
    db = plyvel.DB(dbpath)
    keys = []
    for key, value in db:
        keys.append(key)

    model = train(db, keys)

    model.save('deepdriving_model.h5')
    model.save_weights('deepdriving_weights.h5')
    with open('deepdriving_model.json', 'w') as f:
        f.write(model.to_json())

    db.close()


