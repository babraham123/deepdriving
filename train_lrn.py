from alexnet_lrn import AlexNet
import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import h5py
from keras import backend as K

# nohup python train_lrn.py &
# ps -ef | grep train_lrn.py
# tail -f nohup.out
# kill UID


def train(db, keys, avg):
    m = len(keys)
    # epochs = 19
    # iterations = 140000
    batch_size = 64
    stream_size = batch_size * 100  # ~10K images loaded at a time

    model = AlexNet()
    model.load_weights('alexnet_weights.h5')
    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1, verbose=2)

    # requires adam optimizer
    # model.fit(X_train, Y_train,
    #       batch_size=64, nb_epoch=4700, verbose=1,
    #       validation_data=(X_test, Y_test))
    # max_iter = #epochs * (training set/training_batch_size) 

    return model


def get_data(db, keys, avg):
    n = len(keys)
    if K.image_dim_ordering() == 'tf':
        X_train = np.empty((n, 210, 280, 3))
    else:
        X_train = np.empty((n, 3, 210, 280))

    Y_train = np.empty((n, 14))

    for i, key in enumerate(keys):
        datum = caffe_pb2.Datum.FromString(db.get(key))
        img = caffe.io.datum_to_array(datum)
        # img.shape = 3x210x280
        if K.image_dim_ordering() == 'tf':
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
        # if 'th', leave as is

        img = img.astype('float32')
        img = img / 255
        img = np.subtract(img, avg)
        X_train[i] = img

        affordances = [j for j in datum.float_data]
        affordances = np.array(affordances)
        affordances = affordances.reshape(1, 14)
        affordances = affordances.astype('float32')
        Y_train[i] = affordances

    return X_train, Y_train


def load_average():
    h5f = h5py.File('deepdriving_average.h5', 'r')
    avg = h5f['average'][:]
    h5f.close()
    return avg


def load_keys():
    keys = []
    with open('keys.txt', 'rb') as f:
        keys = [line.strip() for line in f]
    return keys


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    keys = load_keys()

    avg = load_average() 
    model = train(db, keys, avg)

    model.save('deepdriving_model_lrn.h5')
    model.save_weights('deepdriving_weights_lrn.h5')
    #with open('deepdriving_model.json', 'w') as f:
    #    f.write(model.to_json())

    db.close()
