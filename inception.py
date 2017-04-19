from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
import numpy as np
import cv2


# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = '/Users/michael/testdata/train'  # contains two classes cats and dogs
validation_data_dir = '/Users/michael/testdata/validation'

nb_train_samples = 1200
nb_validation_samples = 800
nb_epoch = 50

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(14, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

# train only the top layers (which were randomly initialized)
# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255)#,
 #       shear_range=0.2,
 #       zoom_range=0.2,
 #       horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical'
)

print "start history model"
history = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=128,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples) #1020

