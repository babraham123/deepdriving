#from alexnet_reduced import AlexNet
import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import h5py
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Convolution2D
from keras.layers.normalization import LRN2D
from keras.optimizers import Adam
import itertools
from PIL import Image
from convnet import*
# nenumerate(keys[0,10000])ohup python train.py &
# ps -ef | grep train.py
# kill UID


new_rows = 64
new_cols = 64



def train(db, keys, avg):
    n=len(keys) #32000
    m = len(keys[0:n])
    stream_size = 3200 #batch_size * 100  # ~10K images loaded at a time
   
    batch_size = 32
    
    if K.image_dim_ordering() == 'tf':
        inputs = Input(shape=(new_rows, new_cols, 3))
    else:
        inputs = Input(shape=(3, new_rows, new_cols))

    images = preprocess_image_batch(db, (227, 227), None)
    model =  
    return model


def get_data(db, keys, avg):
    n = len(keys)
    print(n, 'keys')
    if K.image_dim_ordering() == 'tf':
        X_train = np.zeros((n, 64, 64, 3))
    else:
        X_train = np.zeros((n, 3, 64, 64))

    Y_train = np.zeros((n, 14))
    
#    for i, key in enumerate(keys[0,10000]):from PIL import Image
    i=0
    for key in itertools.islice(keys,0,n):
        datum = caffe_pb2.Datum.FromString(db.get(key))
        img = caffe.io.datum_to_array(datum)
        

        # img.shape = 3x210x280
        if K.image_dim_ordering() == 'tf':
            img = img.transpose(1,2,0)
#            print(np.size(img))
        # if 'th', leave as is
        img2 = Image.fromarray(img,'RGB')
        img2 = img2.resize((new_rows, new_cols), Image.ANTIALIAS)
        img =  np.asarray(img2)
#
        img = img.astype('float32')
        img = img / 255
#        img = np.subtract(img, avg)
        X_train[i] = img

        affordances = [j for j in datum.float_data]
        affordances = np.array(affordances)
        affordances = np.expand_dims(affordances,axis=1)
        affordances = affordances.transpose(1,0)
        affordances = affordances.astype('float32')
        Y_train[i] = affordances
        i+=1
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
    
    
    
    
    model.save('a.h5')
    model.save_weights('b.h5')
    with open('c.json', 'w') as f:
        f.write(model.to_json())

    db.close()
    
    
    
    
    
    print('STARTING TESTING')
    #TESTING
    dbpath = '../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/'
    db = plyvel.DB(dbpath)
    keys = []
    for key, value in db:
        keys.append(key)
        
    m = len(keys)
    print(m)
    batch_size = 32
    stream_size = batch_size * 100
    error = np.empty((m, 14))

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        Y_predict = model.predict(X_batch, batch_size=batch_size, verbose=2)
        error[i:(i + stream_size)] = (Y_batch - Y_predict) ** 2
        print(i, 'iteration')
    
    mse = error.mean(axis=0)
    print(mse)
    

