from train import *
import keras 

# nohup python train_board.py &
# tensorboard --logdir /home/asankar/deepdrive/deepdriving/graph 
# http://aiml.me.cmu.edu:6006/
# ps -ef | grep train.py
# kill UID


def train(db, keys, avg):
    m = len(keys)
    # epochs = 19
    # iterations = 140000
    batch_size = 64
    stream_size = batch_size * 100  # ~10K images loaded at a time

    model = AlexNet()
    tbCallback = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    tbCallback.set_model(model)

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1, verbose=1, callbacks=[tbCallback])

    return model


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    keys = []
    for key, value in db:
        keys.append(key)

    avg = load_average()
    model = train(db, keys, avg)

    model.save('deepdriving_model.h5')
    model.save_weights('deepdriving_weights.h5')
    with open('deepdriving_model.json', 'w') as f:
        f.write(model.to_json())

    db.close()
