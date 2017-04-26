import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import plyvel


def convert(db, savepath):
    affordances = []

    for key, val in db:
        key = str(key[2:-1])
        datum = caffe_pb2.Datum.FromString(val)
        img = caffe.io.datum_to_array(datum)
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        cv2.imwrite(savepath + key + '.jpg', img)

        row = [j for j in datum.float_data]
        row.insert(0, key)
        affordances.append(row)

    affordances = np.array(affordances)
    np.save(savepath + 'affordances.npy', affordances)
    # x = np.load(file)


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    savepath = '../train_images/'
    # convert(db, savepath)

    dbpath = '../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/'
    db = plyvel.DB(dbpath)
    savepath = '../test_images_caltech/'
    convert(db, savepath)
