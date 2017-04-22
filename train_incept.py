import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import h5py
from keras import backend as K
from inception import Inception
import cv2
from random import shuffle
import matplotlib.pyplot as plt

# nohup python train.py &
# ps -ef | grep train.py
# tail -f nohup.out
# kill UID

resize = False
normalize = True
random_order = True
plot_loss = True

if K.image_dim_ordering() == 'tf':
    dim = (299, 299, 3)
else:
    dim = (3, 299, 299)


def train_incept(db, keys, avg, mean_std):
    m = len(keys)
    epochs = 20
    # iterations = 140000
    batch_size = 32
    stream_size = batch_size * 100  # ~1K images loaded at a time
    validation_size = batch_size * 10
    loss = []
    val_loss = []

    model = Inception((210, 280, 3), 4096)
    # input shape must be within [139, 299]

    for j in range(epochs):
        for i in range(0, m, stream_size):
            X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg, mean_std)
            X_train = X_batch[:-validation_size]
            Y_train = Y_batch[:-validation_size]
            X_test = X_batch[-validation_size:]
            Y_test = Y_batch[-validation_size:]

            # model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=1)
            hist = model.fit(X_train, Y_train,
                             batch_size=batch_size, epochs=1, verbose=1,
                             validation_data=(X_test, Y_test))
            loss.extend(hist.history['loss'])
            val_loss.extend(hist.history['val_loss'])

    if plot_loss:
        plt.plot(loss)
        plt.plot(val_loss)
        plt.legend(['loss', 'val_loss'])
        plt.savefig('loss_incept.png', bbox_inches='tight')

    return model


def get_data(db, keys, avg, mean_std):
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
        # img = img / 255.0
        if resize:
            img = cv2.resize(img, dim)  # bilinear

        img = np.subtract(img, avg)
        X_train[i] = img

        affordances = [j for j in datum.float_data]
        affordances = np.array(affordances)
        affordances = affordances.reshape(1, 14)
        affordances = affordances.astype('float32')
        if normalize:  # z-score normalization
            affordances = np.subtract(affordances, mean_std[0])
            affordances = np.divide(affordances, mean_std[1])

        Y_train[i] = affordances

    return X_train, Y_train


def calc_output_mean_std(db, keys):
    n = len(keys)
    Y = np.empty((n, 14))
    mean_std = np.empty((2, 14))

    for i, key in enumerate(keys):
        datum = caffe_pb2.Datum.FromString(db.get(key))
        affordances = [j for j in datum.float_data]
        affordances = np.array(affordances)
        affordances = affordances.reshape(1, 14)
        affordances = affordances.astype('float32')
        Y[i] = affordances

    mean_std[0] = np.mean(Y, axis=0)
    mean_std[1] = np.std(Y, axis=0)

    return mean_std


def calc_average(db, keys):
    avg = np.zeros((3, 210, 280))
    n = 0

    for key in keys:
        datum = caffe_pb2.Datum.FromString(db.get(key))
        img = caffe.io.datum_to_array(datum)

        avg = np.add(avg * n, img) / (n + 1)
        n = n + 1

    if K.image_dim_ordering() == 'tf':
        avg = np.swapaxes(avg, 0, 1)
        avg = np.swapaxes(avg, 1, 2)
    # if 'th', leave as is

    avg = avg.astype('float32')
    # avg = avg / 255.0
    return avg


def save_average(avg, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('average', data=avg)
    h5f.close()


def load_average(filename):
    h5f = h5py.File(filename, 'r')
    avg = h5f['average'][:]
    h5f.close()
    return avg


def find_keys(db):
    keys = []
    for key, value in db:
        keys.append(key)
    return keys


def save_keys(keys):
    with open('keys.txt', 'wb') as f:
        f.writelines([b'%s\n' % key for key in keys])


def load_keys():
    keys = []
    with open('keys.txt', 'rb') as f:
        keys = [line.strip() for line in f]
    return keys


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    keys = load_keys()
    avg = load_average('average_no_scale.h5')

    mean_std = calc_average(db, keys)
    save_average(mean_std, 'output_mean_std.h5')
    # mean_std = load_average('output_mean_std.h5')

    if resize:
        avg = cv2.resize(avg, dim)  # bilinear

    if random_order:
        shuffle(keys)

    model = train_incept(db, keys, avg, mean_std)
    model.save('model_inception.h5')

    db.close()

# X -= np.mean(X, axis = 0) # zero-center
# X /= np.std(X, axis = 0) # normalize
