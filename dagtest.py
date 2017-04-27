from gym_torcs2 import TorcsEnv
import numpy as np
from keras.models import model_from_json


img_dim = [64,64,3]

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)
steps = 10000
# load json and create model
json_file = open('Model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights0.h5")
print('here')

ob_list = []
for i in range(steps):
    print('here ag')
    im = img_reshape(ob.img)
    act = loaded_model.predict(im)
    ob, reward, done, _ = env.step(act)
    print('done', done)
    if done is True:
       break
    else:
     ob_list.append(ob)

env.end()

