import caffe
from caffe.proto import caffe_pb2
import plyvel
import numpy as np
import h5py
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout
from keras.models import Model
from convnets import convnet
import itertools
from PIL import Image
import matplotlib.pyplot as plt

# nohup python train.py &
# ps -ef | grep train.py
# kill UID

same_size = True

def train(db, keys, avg):
    m = len(keys[1:100000])

    batch_size = 20
    stream_size = batch_size * 1000  # ~10K images loaded at a time
    K.set_image_dim_ordering('th')
    base_model = convnet('alexnet', weights_path = 'alexnet_weights.h5')
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = Dense(512, activation='relu', init='glorot_normal', name='fc1')(x)
    x = Dense(512, activation='relu', init='glorot_normal', name='fc2')(x)
    #x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', init='glorot_normal', name='fc3')(x)
    x = Dense(14, activation='linear', init='glorot_normal', name='out')(x)

    model = Model(input=base_model.input, output=x)
    model.summary()#AlexNet()

    adam = Adam(lr = 1e-4)
    model.compile(optimizer=adam, loss='mse')
    #sgd = SGD(lr=0.05, decay=0.0005, momentum=0.9)    
    #model.compile(optimizer=sgd, loss='mse')

    for i in range(0, m, stream_size):
        print(i, 'iteration')
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=45, verbose=2)


    return model


def get_data(db, keys, avg):
    n = len(keys)
    if K.image_dim_ordering() == 'tf':
        print('set to tf ordering')
        if not same_size:
            X_train = np.empty((n, 227, 227, 3))
        else:    
            X_train = np.empty((n, 210, 280, 3))
    else:
        print('set to th ordering')
        if not same_size:
            X_train = np.empty((n, 227, 227, 3))
        else:    
            X_train = np.empty((n, 210, 280, 3))
            
    Y_train = np.empty((n, 14))

    for i, key in enumerate(keys):
        datum = caffe_pb2.Datum.FromString(db.get(key))
        img = caffe.io.datum_to_array(datum)


        
        img = img.transpose(1,2,0)
        #print(img.shape)
        img2 = Image.fromarray(img,'RGB')
        
        if not same_size:
            img2 = img2.resize((227, 227), Image.ANTIALIAS)
        
        img =  np.asarray(img2)
        
        img = img.transpose(2,1,0)
        #plt.imshow(img)

        img = img.astype('float32')
        img = img / 255
        #img = np.subtract(img, avg)
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
    dbpath = '/home/asankar/deepdrive/TORCS_Training_1F'
    db = plyvel.DB(dbpath)
    keys = load_keys()

    avg = load_average() 
    model = train(db, keys, avg)

    model.save('deepdriving_model_lrn.h5')
    model.save_weights('deepdriving_weights_lrn.h5')
    #with open('deepdriving_model.json', 'w') as f:
    #    f.write(model.to_json())

    db.close()
