from __future__ import print_function, absolute_import
import numpy as np


from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential #, model_from_json
from keras.regularizers import l2
from keras.layers import Input, Dense, Dropout, Flatten, Activation, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
#import tensorflow as tf
import os
import matplotlib.pyplot as plt

import prepare_dataset as pd

#from keras.utils.generic_utils import get_from_module
#def spatialsoftmax(x):
#    # Assume features is of size [N, H, W, C] (batch_size, height, width, channels).
#    # Transpose it to [N, C, H, W], then reshape to [N * C, H * W] to compute softmax
#    # jointly over the image dimensions. 
#    s = np.shape(x)        
#    x = K.reshape(K.transpose(x, [0, 3, 1, 2]), [s[0] * s[3], s[1] * s[2]])
#    softmax = K.softmax(x)
#    # Reshape and transpose back to original format.
#    softmax = K.transpose(K.reshape(softmax, [s[0], s[3], s[1], s[2]]), [0, 2, 3, 1])        
#    return softmax
#
#def get(identifier):
#    return get_from_module(identifier, globals(), 'activation function') 

model_num = 5### remember to give new model number every iteration
use_mean = False
load_weights = False

#os.path.join(fig_dir, 'model_name' + fig_ext) 
#files_dir = 

logs_path = "/home/exx/Avinash/iMARLO/newdata/run%d/" % model_num
model_filename = 'newdata/model%d.json' % model_num
weights_filename = 'newdata/model%d.h5' % model_num
csvlog_filename = 'newdata/model%d.csv' % model_num

loss_image_filename = 'newdata/model%d_loss.png' % model_num 
pred_image_filename = 'newdata/model%d_predict.png' % model_num    
predsmall_image_filename = 'newdata/model%d_predictsmall.png' % model_num
 
##  tensorboard --logdir /home/exx/Avinash/iMARLO/fullytrained/
tbCallBack = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
csvlog = keras.callbacks.CSVLogger(csvlog_filename, separator=',', append=False )
mdlchkpt = keras.callbacks.ModelCheckpoint(weights_filename, monitor='val_loss',
                 save_best_only=True, save_weights_only=True, period=2)
erlystp = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',min_delta=1e-4,patience=10) 
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.66, patience=10, min_lr=1e-5)


batch_size = 16
nb_epoch = 100
np.random.seed(1337)  # for reproducibility

############################# Prepare Data #########################################

# input image dimensions

Im, cd = pd.get_dataset(regression=True, pretrain = False, image_dir = '/data/new_sl_data')

if K.image_dim_ordering() == 'th':        
    Im = Im/255
    Im = Im.transpose(0,3,1,2)
    Im = Im[:,:,37:131,:] #convert to a square image     
    img_pgs, img_rows, img_cols = Im.shape[1], Im.shape[2], Im.shape[3]
    image_shape = (img_pgs, img_rows, img_cols)
else:    
    Im = Im/255 
    Im = Im[:,:,37:131,:] #convert to a square image
    img_rows, img_cols, img_pgs = Im.shape[1], Im.shape[2], Im.shape[3]
    image_shape = (img_rows, img_cols, img_pgs) 
    
if use_mean:
    ### Subtract Means
    mean_Im = np.mean(Im,axis=0,keepdims=True)
    mean_cd = np.mean(cd,axis=0,keepdims=True)
    Im -= mean_Im
    cd -= mean_cd    

nb_samples = Im.shape[0]
nb_train, nb_test = int(0.75*nb_samples), int(0.25*nb_samples)
### Split Data into test and train
Im_train, Im_test = Im[0:nb_train], Im[nb_train:nb_train + nb_test]
cd_train, cd_test = cd[0:nb_train], cd[nb_train:nb_train + nb_test]

nb_train_small, nb_test_small = int(0.75*nb_train), int(0.75*nb_test)
Im_train_small, Im_test_small = Im_train[0:nb_train_small], Im_test[0:nb_test_small]
cd_train_small, cd_test_small = cd_train[0:nb_train_small], cd_test[0:nb_test_small]

del Im, cd

print('Im_train shape:', Im_train.shape)
print(Im_train.shape[0], 'train samples')
print(Im_test.shape[0], 'test samples')

image_ip = Input(shape=image_shape,name = 'image_input')

randinit = 'glorot_normal'
#if os.path.exists(model_filename):
#    model = pd.get_model_givenpath(model_num, model_filename)
#    if load_weights:
#        model = pd.get_model_givenpath(model_num, model_filename, weights_filename)
#        
#    model.summary()
#    raw_input("Press Enter to continue...")    
#    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss='mse', optimizer='adam', metrics=['mae'])        
#else:        
######################## Load Novel CNN Layers #############################################
print('Loading New CNN model......')

##### DVP Architecture
x = Convolution2D(16, 5, 5, activation='relu', border_mode='valid', init = randinit, name='block1_conv1')(image_ip)
#    x = Convolution2D(16, 5, 5, activation='relu', border_mode='valid', name='block1_conv2')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
#    x = BatchNormalization(16)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init = randinit,  name='block2_conv1')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
#    x = BatchNormalization(32)(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init = randinit,  name='block3_conv1')(x)
   
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init = randinit, name='block4_conv1')(x)

# Block 5
#x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init = 'lecun_uniform', name='block5_conv1')(x)    

# Block 6
#x = Convolution2D(1, 3, 3, activation='relu', border_mode='valid', init = 'lecun_uniform', name='block6_conv1')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#x = GlobalMaxPooling2D()(x)  
x = Flatten()(x)
#x = Activation('softmax')(x)
#    x = BatchNormalization(591680)(x)  
x = Dense(128, activation='relu', init = randinit,  name='fc1')(x)
#    x = BatchNormalization(256)(x)
x = Dense(128, activation='relu', init = randinit,  name='fc2')(x)
#    x = BatchNormalization(256)(x)  
Out = Dense(1,activation='linear', init = randinit,  name='pred')(x)

model = Model(input=image_ip, output=Out)
model.summary()
    
raw_input("Press Enter to continue...")    
adam = Adam(lr=5e-4)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


history = model.fit(Im_train_small, cd_train_small, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, verbose=2,callbacks=[tbCallBack,csvlog,reduce_lr,mdlchkpt])
score_small = model.evaluate(Im_test_small, cd_test_small, verbose=0)
score = model.evaluate(Im_test, cd_test, verbose=0)
print('SmallTestData MSE:', score_small[0])
print('SmallTestData MAE:', score_small[1])
print('FullTestData MSE:', score[0])
print('FullTestData MAE', score[1])

###### Plots ###

eps = history.epoch #np.arange(1,nb_epoch+1,1)
train_loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
    
plt.figure(1)
plt.plot(eps,train_loss,'b-',eps,val_loss,'r-')
plt.ylabel('loss')
plt.xlabel('#epochs')
plt.axis([0, 55, 0, 0.1])
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(loss_image_filename)
print('Plotted Loss for training and validation data')

    
predict_small = model.predict(Im_test_small)


import time
start_time = time.time()

predict = model.predict(Im_test) 


print("Time taken is %s seconds " % (time.time() - start_time))

raw_input("Press Enter to continue...")

if use_mean:     
    # adding mean
    predict+=mean_cd
    cd_test+=mean_cd
    predict_small+=mean_cd
    cd_test_small+=mean_cd
        
    
plt.figure(2)
plt.plot(predict, cd_test,'bo',linewidth=1)
plt.xlabel('Predictions')
plt.ylabel('ActualValue')
plt.title('Learning Evaluation')
plt.grid(True)
plt.savefig(pred_image_filename) 
print('Plotted Predictions for test data')
plt.show()


plt.figure(3)
plt.plot(predict_small, cd_test_small,'bo',linewidth=1)
plt.xlabel('Predictions')
plt.ylabel('ActualValue')
plt.title('Learning Evaluation')
plt.grid(True)
plt.show()
plt.savefig(predsmall_image_filename) 
print('Plotted Predictions for small test data')

model_json = model.to_json()
with open(model_filename, "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")

# serialize weights to HDF5
#model.save_weights(weights_filename) 
#print("Saved weights to disk")

################ modified VGG16 Model ####################
#x = Convolution2D(32, 3, 3, activation='relu', name='block1_conv1')(image_ip)
#x = Convolution2D(32, 3, 3, activation='relu', name='block1_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
#x = Convolution2D(64, 3, 3, activation='relu', name='block2_conv1')(x)
#x = Convolution2D(64, 3, 3, activation='relu', name='block2_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#x = Convolution2D(128, 3, 3, activation='relu', name='block3_conv1')(x)
#x = Convolution2D(128, 3, 3, activation='relu', name='block3_conv2')(x)
#x = Convolution2D(128, 3, 3, activation='relu', name='block3_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#x = Convolution2D(256, 3, 3, activation='relu', name='block4_conv1')(x)
#x = Convolution2D(256, 3, 3, activation='relu', name='block4_conv2')(x)    
#x = Convolution2D(256, 3, 3, activation='relu', name='block4_conv3')(x)

### This reduces to a 1-d array 

#x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

## Block 5
#x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
#x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
#x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

#model = Model(image_ip,x)


########### modified DRVGG16 Model ########################

#x = Convolution2D(64, 3, 3, activation='relu', name='block1_conv1')(image_ip)
#x = Convolution2D(64, 3, 3, activation='relu', name='block1_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
#x = Convolution2D(128, 3, 3, activation='relu', name='block2_conv1')(x)
#x = Convolution2D(128, 3, 3, activation='relu', name='block2_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv1')(x)
#x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv2')(x)
#x = Convolution2D(256, 3, 3, activation='relu', name='block3_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv1')(x)
#x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv2')(x)    
#x = Convolution2D(512, 3, 3, activation='relu', name='block4_conv3')(x)
#
### This reduces to a 1-d array 
#
##x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#
#x = Flatten()(x)    
#x = Dense(256, activation='relu')(x)
#x = Dense(64, activation='relu')(x)
#Out = Dense(1,activation='relu')(x)
#
#model = Model(input=image_ip, output=Out)
#model.summary()
#model = Model(image_ip,x)
