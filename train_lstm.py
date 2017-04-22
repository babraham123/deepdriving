from train_incept import *
from inception_lstm import InceptionLSTM

# nohup python train.py &
# ps -ef | grep train.py
# tail -f nohup.out
# kill UID


def train_lstm(db, keys, avg):
    m = len(keys)
    # epochs = 19
    # iterations = 140000
    batch_size = 32
    stream_size = batch_size * 100  # ~10K images loaded at a time

    model = InceptionLSTM(dim, 4096)

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        # sort into sequences

        model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1, verbose=2)

    return model


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    keys = []
    for key, value in db:
        keys.append(key)

    avg = load_average()
    model = train_lstm(db, keys, avg)

    model.save('deepdriving_model.h5')
    model.save_weights('deepdriving_weights.h5')
    with open('deepdriving_model.json', 'w') as f:
        f.write(model.to_json())

    db.close()
