from alexnet import AlexNet
import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import h5py
from keras.models import load_model

# nohup python evaluate.py &
# ps -ef | grep nohup 
# kill UID 

def evaluate(db, keys, avg):
    m = len(keys)

    model = load_model('deepdriving_model.h5')
    error = np.zeros((0, 14))

    for key in keys:
        datum = caffe_pb2.Datum.FromString(db.get(key))
        img = caffe.io.datum_to_array(datum)
        # img.shape = 3x210x280
        img = img.reshape(1, 3*210*280) / 255
        X = np.subtract(img, avg)
        X = X.astype('float32')

        Y = [i for i in datum.float_data]
        Y = M = np.array(Y)
        Y = Y.reshape(1, 14)
        Y = Y.astype('float32')
        
        Y_predict = model.predict(X)
        error = np.concatenate((error, (Y - Y_predict) ** 2), axis=0)

    mse = error.mean(axis=0)

    return mse


def load_average():
    h5f = h5py.File('deepdriving_average.h5','r')
    avg = h5f['average'][:]
    h5f.close()
    return avg


if __name__ == "__main__":
    dbpath = '../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/'
    db = plyvel.DB(dbpath)
    keys = []
    for key, value in db:
        keys.append(key)

    avg = load_average()
    scores = evaluate(db, keys, avg)
    print(scores)

    db.close()

# datum.add_float_data(shared->angle);
# datum.add_float_data(shared->toMarking_L);
# datum.add_float_data(shared->toMarking_M);
# datum.add_float_data(shared->toMarking_R);
# datum.add_float_data(shared->dist_L);
# datum.add_float_data(shared->dist_R);
# datum.add_float_data(shared->toMarking_LL);
# datum.add_float_data(shared->toMarking_ML);
# datum.add_float_data(shared->toMarking_MR);
# datum.add_float_data(shared->toMarking_RR);
# datum.add_float_data(shared->dist_LL);
# datum.add_float_data(shared->dist_MM);
# datum.add_float_data(shared->dist_RR);
# datum.add_float_data(shared->fast);