from gym_torcs import TorcsEnv
from controller import*
import numpy as np

img_dim = [64,64,3]
action_dim = 3
steps = 1000
batch_size = 32
nb_epoch = 100

def get_teacher_action(ob):
    steer = ob.angle*10/np.pi
    steer -= ob.trackPos*0.10
    return np.array([steer])

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0,action_dim))
rewards_all = np.zeros((0,))

img_list = []
action_list = []
reward_list = []

env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)

print('Collecting data...')
for i in range(steps):
    if i == 0:
        act = np.array([0.0])
    else:
        act = get_teacher_action(ob)

    if i%100 == 0:
        print(i)
    ob, reward, done, _ = env.step(act)
    img_list.append(ob.img)
    action_list.append(act)
    reward_list.append(np.array([reward]))

env.end()

print('Packing data into arrays...')
for img, act, rew in zip(img_list, action_list, reward_list):
    images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
    actions_all = np.concatenate([actions_all, np.reshape(act, [1,action_dim])], axis=0)
    rewards_all = np.concatenate([rewards_all, rew], axis=0)

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from controller.py import*
#model from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
# load json and create model
#json_file = open('simplified_mdl_lrn.h5', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
loaded_model = load_model('simplified_mdl_lrn.h5')
# load weights into new model
loaded_model.load_weights("simplified_wts_lrn.h5")

model.fit(images_all, actions_all,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True)

output_file = open('results.txt', 'w')

#aggregate and retrain
dagger_itr = 5
for itr in range(dagger_itr):
    ob_list = []

    env = TorcsEnv(vision=True, throttle=False)
    ob = env.reset(relaunch=True)
    reward_sum = 0.0
    raw_act = model.predict(img_reshape(ob.img))
    prev_act = raw_act ###FIX THIS
    for i in range(steps):
        raw_act = model.predict(img_reshape(ob.img))
                
        ob, reward, done, _ = env.step(act)
        if done is True:
            break
        else:
            ob_list.append(ob)
        reward_sum += reward
        print(i, reward, reward_sum, done, str(act[0]))
    print('Episode done ', itr, i, reward_sum)
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(i, reward_sum))
    env.end()

    if i==(steps-1):
        break

    for ob in ob_list:
        images_all = np.concatenate([images_all, img_reshape(ob.img)], axis=0)
        actions_all = np.concatenate([actions_all, np.reshape(get_teacher_action(ob), [1,action_dim])], axis=0)

    model.fit(images_all, actions_all,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True)
