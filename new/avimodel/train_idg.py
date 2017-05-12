from keras.models import Model, model_from_json
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Reshape, Permute, Activation, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
import numpy as np
import h5py
import cv2
from glob import glob
from time import time
from os.path import isfile
from random import shuffle
import matplotlib.pyplot as plt
start_time = time()

# source activate deepenv1
# nohup python train.py &
# ps -ef | grep train.py
# kill UID


def train(db, keys, avg, batch_size, epochs, nb_tr, nb_val , samples=None, val_samples=None, labels=True, scale_affords= False):

    if samples is None:
        samples = int(nb_tr/batch_size)
    if val_samples is None:
        val_samples = int(nb_val/batch_size)

    if pretrained and isfile(weights_filename):
        model = alexnet(weights_path=weights_filename)
    else:
        model = alexnet()

    model.fit_generator( our_datagen(db, keys[0:nb_tr], avg, batch_size, labels=True,scale_affords=scale_out),
        samples_per_epoch = samples, nb_epoch = epochs,
        verbose=2, callbacks=[csvlog, reduce_lr, mdlchkpt,tbCallBack],
         validation_data=our_datagen(db, keys[nb_tr:nb_tr+nb_val], avg, batch_size, labels=True,scale_affords=scale_out),
         nb_val_samples=val_samples)

    model.save(model_filename)
    model.save_weights(weights_filename)
    return model


def alexnet(weights_path=None):
    """
    Returns a keras model for a CNN.
    input data are of the shape (227,227), and the colors in the RGB order (default)

    model: The keras model for this convnet
    output_dict: Dict of feature layers, asked for in output_layers.
    """
    inputs = Input(shape=dim)

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu', name='conv_1')(inputs)
    # initial weights filler? gaussian, std 0.01
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    # in caffe: Local Response Normalization (LRN)

    # alpha = 1e-4, k=2, beta=0.75, n=5,
    #conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = Convolution2D(256, 5, 5, activation="relu", name='conv_2')(conv_2)

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    
    #conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    #conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Convolution2D(384, 3, 3, activation="relu", name='conv_4')(conv_3)

    #conv_4 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Convolution2D(256, 3, 3, activation="relu", name='conv_5')(conv_4)
    
    if same_size is True:
        dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)
        dense_1 = Flatten(name="flatten")(dense_1)
    else:    
        dense_1 = Flatten(name="flatten")(conv_5)#(dense_1)
    
    # initial weights filler? gaussian, std 0.005
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)

    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)

    # initial weights filler? gaussian, std 0.01
    dense_3 = Dense(256, activation='relu', name='dense_3')(dense_3)
    dense_4 = Dropout(0.5)(dense_3)

    # output: 14 affordances, gaussian std 0.01
    dense_4 = Dense(13, activation='linear', name='dense_4')(dense_4)
    # dense_4 = Dense(14, activation='linear', name='dense_4')(dense_4)

    model = Model(input=inputs, output=dense_4)
    model.summary()
    raw_input("Press Enter to continue...")  
    
    if weights_path:
        model.load_weights(weights_path)

    # sgd = SGD(lr=0.01, decay=0.0005, momentum=0.9)  # nesterov=True) # LSTM
    adam = Adam(lr=5e-4)
    model.compile(optimizer=adam, loss='mse',metrics=['mae'])  # try cross-entropy

    return model


def our_datagen(db, keys, avg,batch_size,labels=True,scale_affords=False):
    n = len(keys)/batch_size
    n = int(n)
    affordance_dim = 13
    for index in range(0,n):
        xdim = (batch_size,) + dim
        X_train = np.zeros(xdim)
        Y_train = np.zeros((batch_size, affordance_dim))

        for i, key in enumerate(keys[index:(index+batch_size)]):
            img = cv2.imread(key)
            # img.shape = 210x280x3
            if not same_size:
                img = cv2.resize(img, (64, 64))

            img = img / 255.0
            img = np.subtract(img, avg)
            if K.image_dim_ordering() == 'th':
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 0, 1)

            X_train[i] = img
            
            if labels is True:
                j = int(key[-12:-4])
                affordances = db[j - 1]
                if int(affordances[0]) != j:
                    raise ValueError('Image and affordance do not match: ' + str(j))
                affordances = affordances[1:(affordance_dim+1)]
                if scale_affords is True:
                    affordances = scale_output(affordances)
                affordances = affordances.reshape(1, affordance_dim)
                Y_train[i] = affordances
            
            if labels is True:
                yield X_train, Y_train
            else:
                yield X_train
                
def predict_affordances(db, keys, avg, model, batch_size, verbose = 0, scale_affords=False):
    nb_ts = len(keys)
    nb = int(nb_ts/batch_size)    
    affordance_dim = 13
    Y_true = np.zeros((nb*batch_size, affordance_dim))
    Y_pred = np.zeros((nb*batch_size, affordance_dim))
    err = np.zeros((nb*batch_size, affordance_dim))
    err_avg = np.zeros((1, affordance_dim))
    
    for index in range(0,nb):
        #xdim = (batch_size,) + dim
        #X_train = np.zeros(xdim)
        #Y_train = np.zeros((batch_size, affordance_dim))

        for i, key in enumerate(keys[index:(index+batch_size)]):
            img = cv2.imread(key)
            # img.shape = 210x280x3
            if not same_size:
                img = cv2.resize(img, (64, 64))

            img = img / 255.0
            img = np.subtract(img, avg)
            if K.image_dim_ordering() == 'th':
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 0, 1)

            img = np.expand_dims(img, axis=0)
            
            j = int(key[-12:-4])
            affordances = db[j - 1]
            if int(affordances[0]) != j:
                raise ValueError('Image and affordance do not match: ' + str(j))
            affordances = affordances[1:(affordance_dim+1)]
            if scale_affords is True:
                affordances = scale_output(affordances)
            affordances = affordances.reshape(1, affordance_dim)
            affords_pred = model.predict(img)
            Y_true[i + (index*batch_size)] = affordances
            Y_pred[i + (index*batch_size)] = affords_pred
            err[i + (index*batch_size)] = np.abs(affords_pred - affordances)
        #predict.append(Y_train)
        if verbose is 1:
            test = (index+1)*batch_size
            print('Number of samples predicted so far:' + str(test))
   
    err_avg = err.mean(axis=0)   
        
    return Y_pred, Y_true, err, err_avg
                

def scale_output(affordances):
    ''' Scale output between [0.1, 0.9]
    '''
    affordances[0] = affordances[0] / 1.1 + 0.5         # angle

    affordances[1] = affordances[1] / 5.6249 + 1.34445  # toMarking_L
    affordances[2] = affordances[2] / 6.8752 + 0.39091  # toMarking_M
    affordances[3] = affordances[3] / 5.6249 - 0.34445  # toMarking_R

    affordances[4] = affordances[4] / 95 + 0.12         # dist_L
    affordances[5] = affordances[5] / 95 + 0.12         # dist_R

    affordances[6] = affordances[6] / 6.8752 + 1.48181  # toMarking_LL
    affordances[7] = affordances[7] / 6.25 + 0.98       # toMarking_ML
    affordances[8] = affordances[8] / 6.25 + 0.02       # toMarking_MR
    affordances[9] = affordances[9] / 6.8752 - 0.48181  # toMarking_RR

    affordances[10] = affordances[10] / 95 + 0.12       # dist_LL
    affordances[11] = affordances[11] / 95 + 0.12       # dist_MM
    affordances[12] = affordances[12] / 95 + 0.12       # dist_RR
    return affordances


def descale_output(affordances): 
    affordances_unnorm = np.zeros(affordances.shape)    
    
    affordances_unnorm[:,0] = (affordances[:,0] - 0.5) * 1.1

    affordances_unnorm[:,1] = (affordances[:,1] - 1.34445) * 5.6249
    affordances_unnorm[:,2] = (affordances[:,2] - 0.39091) * 6.8752
    affordances_unnorm[:,3] = (affordances[:,3] + 0.34445) * 5.6249

    affordances_unnorm[:,4] = (affordances[:,4] - 0.12) * 95
    affordances_unnorm[:,5] = (affordances[:,5] - 0.12) * 95

    affordances_unnorm[:,6] = (affordances[:,6] - 1.48181) * 6.8752
    affordances_unnorm[:,7] = (affordances[:,7] - 0.98) * 6.25
    affordances_unnorm[:,8] = (affordances[:,8] - 0.02) * 6.25
    affordances_unnorm[:,9] = (affordances[:,9] + 0.48181) * 6.8752

    affordances_unnorm[:,10] = (affordances[:,10] - 0.12) * 95
    affordances_unnorm[:,11] = (affordances[:,11] - 0.12) * 95
    affordances_unnorm[:,12] = (affordances[:,12] - 0.12) * 95
    return affordances_unnorm

def load_average():
    h5f = h5py.File('/home/exx/Avinash/DReD/local/deepdriving_average.h5', 'r')
    avg = h5f['average'][:]
    h5f.close()
    return avg


if __name__ == "__main__":
    dbpath = '/data/deepdriving/train_images/'
    
    keys = glob(dbpath + '*.jpg')
    
    #keys.sort()
    
    
    db = np.load(dbpath + 'affordances.npy')

    # TODO : shuffle and keep aligned

    db = db.astype('float32')

    avg = load_average()
        
    scale_out = False    
    same_size = True
    pretrained = False
    model_num = 9
    folder = "/home/exx/Avinash/DReD/local/"
    
    model_filename = folder + 'models/cnnmodel%d.json' % model_num
    weights_filename = folder + 'models/cnnmodel%d_weights.h5' % model_num
    
    logs_path = folder + "models/run%d/" % model_num
    csvlog_filename = folder + 'models/cnnmodel%d.csv' % model_num
    
    #  tensorboard --logdir /home/exx/Avinash/DReD/local/models/
    tbCallBack = TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)
    csvlog = CSVLogger(csvlog_filename, separator=',', append=False)
    mdlchkpt = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=True, period=2, verbose=1)
    erlystp = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5, verbose=1)
    
    if K.image_dim_ordering() == 'tf':
        print('Tensorflow')
        if same_size:
            dim = (210, 280, 3)
        else:
            dim = (64, 64, 3)
    else:
        print('Theano')
        if same_size:
            dim = (3, 210, 280)
        else:
            dim = (3, 64, 64)

    # avg.shape = 210x280x3
    if not same_size:
        avg = cv2.resize(avg, (64, 64))

    batch_size = 32
    epochs = 25
    nb_tr = 350000
    nb_val = 50000
    nb_ts = 5056 #84800
    if os.path.exists(model_filename):
        json_file = open(model_filename, 'r')
        model_json = json_file.read()
        json_file.close()
        print('Model found and loading ...')        
        model = model_from_json(model_json)
        print("Loading the best weights for evaluation")                
        model.load_weights(weights_filename)  
        adam = Adam(lr=5e-4)
        model.compile(optimizer=adam, loss='mse',metrics=['mae'])  # try cross-entropy                  
    else:
        print('New model is built and training...')
        model = train(db, keys, avg, batch_size, epochs, nb_tr, nb_val , samples=None, val_samples=None, labels=True,scale_affords=scale_out)
        print("Loading the best weights for evaluation")
        model.load_weights(weights_filename)   
        
        # saving the model to disk        
        model_json = model.to_json()
        with open(model_filename, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")
    
    ts_samples = int(nb_ts/batch_size)
    score = model.evaluate_generator(our_datagen(db, keys[nb_tr+nb_val:nb_tr+nb_val+nb_ts], avg, batch_size), ts_samples) 
    print('TestData MSE:', score[0])
    print('TestData MAE', score[1])
    
    Y_pred, Y_true, err, err_avg = predict_affordances(db, keys[nb_tr+nb_val:nb_tr+nb_val+nb_ts], avg, model, batch_size, verbose=1, scale_affords = scale_out)
   
    if scale_out is True:
        Y_pred_unnorm = descale_output(Y_pred)
        Y_true_unnorm = descale_output(Y_true)
        err = descale_output(err)    
        err_avg = descale_output(err_avg.reshape(1,13))
    
    
    print("Time taken is %s seconds " % (time() - start_time))
