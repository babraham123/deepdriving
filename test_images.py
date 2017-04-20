import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import plyvel


def load_keys():
    keys = []
    with open('keys.txt', 'rb') as f:
        keys = [line.strip() for line in f]
    return keys


if __name__ == "__main__":
    keys = load_keys()
    keys.sort()

    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)

    savepath = '../train_images/'

    for i in range(1000):
        key = keys[i]
        datum = caffe_pb2.Datum.FromString(db.get(key))
        img = caffe.io.datum_to_array(datum)
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)

        cv2.imwrite(savepath + 'img' + str(key) + '.jpg', img)
