from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


def Inception():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = GlobalMaxPooling2D()(x)

    x = Dense(400, activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    # 4096

    predictions = Dense(14, activation='sigmoid')(x)
    model = Model(input=base_model.input, output=predictions)

    # train only the top layers (which were randomly initialized)
    # freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # sgd = SGD(lr=0.01, decay=0.0005, momentum=0.9) # nesterov=True)
    # model.compile(optimizer=sgd, loss='mean_squared_error')

    model.compile(loss='mse', optimizer='adam')

    return model