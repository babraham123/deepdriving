from train_combined import *

# nohup python evaluate.py &
# ps -ef | grep evaluate.py
# tail -f nohup.out
# kill UID


def test_data(db, keys, avg):
    m = len(keys)
    batch_size = 16
    stream_size = batch_size * 500  # ~10K images loaded at a time

    Y = np.empty((m, 14))
    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)

        Y[i:(i + stream_size)] = Y_batch

    mean = Y.mean(axis=0)
    maxx = Y.amax(axis=0)
    minn = Y.amin(axis=0)
    std = Y.std(axis=0)
    return mean[display_idx], maxx[display_idx], minn[display_idx], std[display_idx]


if __name__ == "__main__":
    dbpath = '/home/lkara/deepdrive/train_images/'
    keys = glob(dbpath + '*.jpg')
    keys.sort()
    db = np.load(dbpath + 'affordances.npy')
    db = db.astype('float32')

    avg = load_average()
    # avg.shape = 210x280x3
    if not same_size:
        avg = cv2.resize(avg, (227, 227))

    s1, s2, s3, s4 = test_data(db, keys, avg)
    print('Mean: ' + str(s1))
    print('Std: ' + str(s4))
    print('Max: ' + str(s2))
    print('Min: ' + str(s3))
    print("Time taken is %s seconds " % (time() - start_time))
