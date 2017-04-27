from train_alexnet import *
from keras.models import load_model

# nohup python evaluate.py &
# ps -ef | grep evaluate.py
# tail -f nohup.out
# kill UID


def evaluate(db, keys, avg):
    m = len(keys)
    batch_size = 16
    stream_size = batch_size * 500  # ~10K images loaded at a time

    model = load_model('alexnet1.h5')

    error = np.empty((m, 14))

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        Y_predict = model.predict(X_batch, batch_size=batch_size, verbose=1)
        error[i:(i + stream_size)] = (Y_batch - Y_predict) ** 2

    mse = error.mean(axis=0)
    return mse


if __name__ == "__main__":
    dbpath = '/home/lkara/deepdrive/test_images_caltech/'
    keys = glob(dbpath + '*.jpg')
    keys.sort()
    db = np.load(dbpath + 'affordances.npy')
    db = db.astype('float32')

    avg = load_average()
    # avg.shape = 210x280x3
    if not same_size:
        avg = cv2.resize(avg, (227, 227))

    scores = evaluate(db, keys, avg)
    print(scores)
    print("Time taken is %s seconds " % (time() - start_time))


# angle);
# toMarking_L);
# toMarking_M);
# toMarking_R);
# dist_L);
# dist_R);
# toMarking_LL);
# toMarking_ML);
# toMarking_MR);
# toMarking_RR);
# dist_LL);
# dist_MM);
# dist_RR);
# fast);
