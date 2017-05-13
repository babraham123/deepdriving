from gym_torcs2 import TorcsEnv
import numpy as np
from keras.models import model_from_json, load_model
import cv2
from controller import*

img_dim = [210,280,3]

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = cv2.resize(_img, img_dim[0], img_dim[1])
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

# load json and create model
'''
json_file = open('Model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
'''
#load model from h5
loaded_model = load_model('new/avimodels')
# load weights into new model
#loaded_model.load_weights("weights0.h5")
print('Loaded model')


#Start TORCS environment
env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)
steps = 10000


ob_list = []
aff_list = []
for i in range(steps):
    print('Step:'. i)
    im = img_reshape(ob.img)
    affordances = loaded_model.predict(im)
    if i< 1:
        aff_list.append(affordances)
        continue

    #Give controller affordance, past aff, steer record, and state    
    conv_act = controller(affordances, aff_list[i],steer_record, ob)
    
    ob, reward, done, _ = env.step(conv_act)
    print('Taken action')
    if done is True:
       break
    else:
       ob_list.append(ob)
       aff_list.append(affordances)
       


env.end()

