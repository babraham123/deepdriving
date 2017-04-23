from train_incept import *
from inception_lstm import InceptionLSTM

# nohup python train.py &
# ps -ef | grep train.py
# tail -f nohup.out
# kill UID


def train_lstm(db, keys, avg, mean_std):
    m = len(keys)
    epochs = 20
    # iterations = 140000
    batch_size = 32
    stream_size = batch_size * 100  # ~1K images loaded at a time
    validation_size = batch_size * 10
    loss = []
    val_loss = []

    model = InceptionLSTM((210, 280, 3), 4096)
    # input shape must be within [139, 299]

    # TODO: break keys up into sequences

    for j in range(epochs):
        for i in range(0, m, stream_size):
            X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg, mean_std)
            X_train = X_batch[:-validation_size]
            Y_train = Y_batch[:-validation_size]
            X_test = X_batch[-validation_size:]
            Y_test = Y_batch[-validation_size:]

            # model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1, verbose=1)
            hist = model.fit(X_train, Y_train,
                             batch_size=batch_size, epochs=1, verbose=1,
                             validation_data=(X_test, Y_test))
            loss.extend(hist.history['loss'])
            val_loss.extend(hist.history['val_loss'])

    if plot_loss:
        plt.plot(loss)
        plt.plot(val_loss)
        plt.legend(['loss', 'val_loss'])
        plt.savefig('loss_incept.png', bbox_inches='tight')

    return model


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    keys = load_keys()
    avg = load_average('average_no_scale.h5')
    mean_std = load_average('output_mean_std.h5')

    model = train_lstm(db, keys, avg, mean_std)
    model.save('deepdriving_model.h5')

    db.close()
