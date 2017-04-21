from train import *
from inception import Inception
import cv2

# nohup python train.py &
# ps -ef | grep train.py
# tail -f nohup.out
# kill UID


def train2(db, keys, avg):
    m = len(keys)
    # epochs = 19
    # iterations = 140000
    batch_size = 64
    stream_size = batch_size * 20  # ~1K images loaded at a time

    model = Inception((210, 280, 3))
    # input shape must be within [139, 299]

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=2)

    return model


def get_data_resize(db, keys, avg):
    n = len(keys)
    if K.image_dim_ordering() == 'tf':
        X_train = np.empty((n, 210, 280, 3))
        dim = (299, 299, 3)
    else:
        X_train = np.empty((n, 3, 210, 280))
        dim = (3, 299, 299)

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
        img = img / 255.0
        # img = cv2.resize(img, dim)  # bilinear
        img = np.subtract(img, avg)
        X_train[i] = img

        affordances = [j for j in datum.float_data]
        affordances = np.array(affordances)
        affordances = affordances.reshape(1, 14)
        affordances = affordances.astype('float32')
        Y_train[i] = affordances


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    keys = load_keys()

    avg = load_average()
    # avg = cv2.resize(avg, (299, 299, 3))  # bilinear
    model = train2(db, keys, avg)

    model.save('deepdriving_model_inception.h5')

    db.close()
