import numpy as np
from keras import backend as K
from keras.models import load_model
import h5py


class Agent(object):
    def __init__(self, dim_action):
        self.model = load_model('deepdriving_model_rand.h5')
        self.average = self.load_average()
        self.is_tf = (K.image_dim_ordering() == 'tf')

    def load_average():
        h5f = h5py.File('deepdriving_average.h5', 'r')
        avg = h5f['average'][:]
        h5f.close()
        return avg

    def preprocess_image(self, vision):
        im = np.resize(vision, (210, 280, 3))
        # crop, zero pad, etc 
        if self.is_tf is False:
            im = np.swapaxes(im, 0, 1)
        return im

    def controller(self, affordances):
        action = [0.0, 0.0]
        return action

    def act(self, ob, reward, done, vision_on):
        '''
        Get an Observation from the environment.
        Each observation vectors are numpy array.
        focus, opponents, track sensors are scaled into [0, 1]. When the agent
        is out of the road, sensor variables return -1/200.
        rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        and values are in [0, 255]
        '''

        if vision_on:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision = ob
            '''
            The code below is for checking the vision input. 
            This is very heavy for real-time Control
            So you may need to remove.
            
            img = np.ndarray((64,64,3))
            for i in range(3):
                img[:, :, i] = 255 - vision[:, i].reshape((64, 64))
            plt.imshow(img, origin='lower')
            plt.draw()
            plt.pause(0.001)
            '''

            im = self.preprocess_image(vision)
            affordances = self.model.predict(im)
            action = self.controller(affordances)

        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel = ob
            action = np.tanh(np.random.randn(2))  # random action

        # action = (steering angle, throttle amt)
        return action
