from train_rnn import *
from keras.models import load_model

# nohup python evaluate.py &
# ps -ef | grep evaluate.py
# tail -f nohup.out
# kill UID


def evaluate(db, keys, avg):
    m = len(keys)
    batch_size = 1  # 16
    stream_size = batch_size * 100  # ~10K images loaded at a time
    hist_size = 4

    try:
        model = load_model(folder + "models/alexnet%d.h5" % model_num)
    except Exception:
        model = alexnet()
        model.load_weights(folder + "models/model_weights%d.h5" % model_num)

    n = ((stream_size - hist_size + 1) * floor(m / stream_size)) + ((m % stream_size) - hist_size + 1)
    error = np.empty((n, 14))
    error2 = np.empty((n, 14))

    j = 0
    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        X_batch, Y_batch = convert_sequence_sliding(X_batch, Y_batch, hist_size)
        Y_batch = Y_batch[:, hist_size - 1]
        num_seq = Y_batch.shape[0]
        #score = model.evaluate(X_batch,Y_batch,verbose=0)
        #mae = score[1] 
        #mse = score[0] 

        Y_predict = model.predict(X_batch, batch_size=batch_size, verbose=1)
        Y_predict = Y_predict[:, hist_size - 1]
        for k in range(Y_predict.shape[0]):
            Y_predict[k] = descale_output(Y_predict[k])

        error[j:(j + num_seq)] = np.absolute(Y_batch - Y_predict)
        error2[j:(j + num_seq)] = np.square(Y_batch - Y_predict)
        j += num_seq

    if(j != n):
        raise ValueError('Number of sequences do not match: ' + str(j) + ' ' + str(n))

    mae = error.mean(axis=0)
    mse = error2.mean(axis=0)
    return mae, mse


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

    scores, scores2 = evaluate(db, keys, avg)
    print('Mean absolute error: ' + str(scores))
    print('Mean squared error: ' + str(scores2))
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

