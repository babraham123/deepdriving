from train import *
from keras.models import load_model, model_from_json

# nohup python evaluate.py &
# ps -ef | grep evaluate.py
# tail -f nohup.out
# kill UID


def evaluate(db, keys, avg):
    m = len(keys)
    # epochs = 19
    # iterations = 140000
    batch_size = 32
    stream_size = batch_size * 100  # ~10K images loaded at a time
    
    print('before')
    try:
        model = load_model('a.h5')
        print('h5')
    except:
        json_file = open('c.json', 'r')
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights('simplified_wts_lrn_redo.h5')
        print('here')
        
#    model = load_model('simplified_mdl_lrn_redo.h5')
    
    error = np.empty((m, 14))

    for i in range(0, m, stream_size):
        X_batch, Y_batch = get_data(db, keys[i:(i + stream_size)], avg)
        print(X_batch.shape)
        Y_predict = model.predict(X_batch, batch_size=batch_size, verbose=1)
        error[i:(i + stream_size)] = (Y_batch - Y_predict) ** 2

    mse = error.mean(axis=0)
    return mse


if __name__ == "__main__":
    dbpath = '../TORCS_baseline_testset/TORCS_Caltech_1F_Testing_280/'
    db = plyvel.DB(dbpath)
    keys = load_keys()

    avg = load_average()
    scores = evaluate(db, keys, avg)
    print(scores)

    db.close()

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
