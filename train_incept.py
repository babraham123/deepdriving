import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import h5py
from keras import backend as K
from inception import Inception
import cv2

# nohup python train.py &
# ps -ef | grep train.py
# tail -f nohup.out
# kill UID

resize = False
if K.image_dim_ordering() == 'tf':
    dim = (299, 299, 3)
else:
    dim = (3, 299, 299)


def train_incept(db, keys, avg):
    m = len(keys)
    # epochs = 19
    # iterations = 140000
    batch_size = 32
    stream_size = batch_size * 100  # ~1K images loaded at a time

    model = Inception((210, 280, 3))
    # input shape must be within [139, 299]

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=2)

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
        # img = img / 255.0
        if resize:
            img = cv2.resize(img, dim)  # bilinear

        img = np.subtract(img, avg)
        X_train[i] = img

        affordances = [j for j in datum.float_data]
        affordances = np.array(affordances)
        affordances = affordances.reshape(1, 14)
        affordances = affordances.astype('float32')

        # PCA whitening
        # TODO::
        Y_train[i] = affordances

    return X_train, Y_train


def calc_output_mean_var(db, keys):
    n = len(keys)
    Y = np.empty((n, 14))
    mean_var = np.empty((2, 14))

    for i, key in enumerate(keys):
        datum = caffe_pb2.Datum.FromString(db.get(key))
        affordances = [j for j in datum.float_data]
        affordances = np.array(affordances)
        affordances = affordances.reshape(1, 14)
        affordances = affordances.astype('float32')
        Y[i] = affordances

    mean_var[0] = np.mean(Y, axis=0)
    mean_var[1] = np.var(Y, axis=0)

    return mean_var


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
    # mean_var = load_average('output_mean_var.h5')

    if resize:
        avg = cv2.resize(avg, dim)  # bilinear

    model = train_incept(db, keys, avg)
    model.save('model_inception.h5')

    db.close()

# output_mean_var
# [[  1.05096394e-03  -6.10439122e+00   2.22679069e+00   6.01857480e+00
#     6.55837606e+01   6.50148110e+01  -7.62196768e+00  -2.40765018e+00
#     2.51720775e+00   7.66177475e+00   5.68282719e+01   4.57264860e+01
#     5.64878142e+01   2.80581253e-01]
#  [  1.10733842e-02   2.08471159e+00   3.13468153e+00   2.30695793e+00
#     3.50281351e+02   3.65920679e+02   3.50163140e+00   1.85120518e+00
#     1.74987999e+00   3.30746772e+00   5.80191632e+02   5.42373414e+02
#     5.90709864e+02   2.01855413e-01]]
