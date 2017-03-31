from train import *
import random

# nohup python train_rand.py &
# ps -ef | grep train_rand.py
# kill UID


def train_rand(db, keys, avg):
    iterations = 140000
    batch_size = 64

    model = AlexNet()

    for i in range(iterations):
        batch = random.sample(keys, batch_size)
        X_batch, Y_batch = get_data(db, batch, avg)
        model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1, verbose=1)

    return model


if __name__ == "__main__":
    dbpath = '../TORCS_Training_1F/'
    db = plyvel.DB(dbpath)
    keys = []
    for key, value in db:
        keys.append(key)

    avg = load_average()
    model = train_rand(db, keys, avg)

    model.save('deepdriving_model_rand.h5')
    model.save_weights('deepdriving_weights_rand.h5')

    db.close()
