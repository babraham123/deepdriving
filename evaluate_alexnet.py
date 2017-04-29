from train_alexnet_normalized import *
from keras.models import load_model, load_weights

# nohup python evaluate.py &
# ps -ef | grep evaluate.py
# tail -f nohup.out
# kill UID


def evaluate(db, keys, avg):
    m = len(keys)
    batch_size = 16
    stream_size = batch_size * 500  # ~10K images loaded at a time

    try:
        model = load_model('models/model1.h5')
    except Exception:
        model = get_model()
        model.load_weights('models/model_weights1.h5')
        

    error = np.empty((m, 14))

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        Y_predict = model.predict(X_batch, batch_size=batch_size, verbose=1)
        Y_predict = descale_output(Y_predict)
        
        error[i:(i + stream_size)] = np.absolute(Y_batch - Y_predict)

    mae = error.mean(axis=0)
    return mae


if __name__ == "__main__":
    dbpath = '/home/lkara/deepdrive/test_images_gist/'
    keys = glob(dbpath + '*.jpg')
    keys.sort()
    db = np.load(dbpath + 'affordances.npy')
    db = db.astype('float32')

    avg = load_average()
    # avg.shape = 210x280x3
    if not same_size:
        avg = cv2.resize(avg, (227, 227))

    scores = evaluate(db, keys, avg)
    print('Mean absolute error: ' + str(scores))
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
