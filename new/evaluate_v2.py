from keras.models import model_from_json

# source activate deepenv1
# nohup python evaluate.py &
# ps -ef | grep evaluate.py
# tail -f nohup.out
# kill UID
def test_data(db, keys, avg):


    img = cv2.imread(key)
    if not same_size:
        img = cv2.resize(img, (227, 227))

    img = img.astype('float32')
    img = img / 255.0
    img = np.subtract(img, avg)
    if K.image_dim_ordering() == 'th':
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)

    X_train= img

    return X_train

def evaluate(db, keys, avg):
    m = len(keys)
    batch_size = 1  # 16
    stream_size = batch_size * 500  # ~10K images loaded at a time

    model_name = "avimodel/cnnmodel9.json"



    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("avimodel/cnnmodel9_weights.h5")

    error = np.empty((m, 14))
    error2 = np.empty((m, 14))

    for i in range(0, m, stream_size):
        X_batch, Y_batch = test_data(db, keys[i:(i + stream_size)], avg)

        Y_predict = model.predict(X_batch, batch_size=batch_size, verbose=1)
        for k in range(Y_predict.shape[0]):
            Y_predict[k] = descale_output(Y_predict[k])

        error[i:(i + stream_size)] = np.absolute(Y_batch - Y_predict)
        error2[i:(i + stream_size)] = np.square(Y_batch - Y_predict)

    mae = error.mean(axis=0)
    mse = error2.mean(axis=0)
    return mae[display_idx], mse[display_idx]


if __name__ == "__main__":
    dbpath = '/home/lkara/deepdrive/test_images_caltech/'
    # dbpath = '/home/lkara/deepdrive/test_images_gist/'
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

