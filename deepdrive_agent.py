import numpy as np
from keras import backend as K
from keras.models import load_model
import h5py
from controller import controller


class Agent(object):
    def __init__(self, dim_action):
        self.model = load_model('deepdriving_model_rand.h5')
        self.average = self.load_average()
        self.is_tf = (K.image_dim_ordering() == 'tf')
        self.steering_record = [0, 0, 0, 0, 0]
        self.prev_affordances = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30.0, 0, 30.0, 0]

        road_width = 8.0
        coe_steer = 1.0
        lane_change = 0
        steering_head = 0

        left_clear = 0
        right_clear = 0
        left_timer = 0
        right_timer = 0
        timer_set = 60
        pre_dist_L = 60.0
        pre_dist_R = 60.0
        steer_trend
        goto_lane = 0

        self.state = [road_width, steering_head, timer_set, lane_change, speed, goto_lane]
        # road_width, steering_head, timer_set, lane_change, speed = state

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
            action = controller(affordances, self.prev_affordances, self.state)
            self.prev_affordances = affordances

        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel = ob
            action = np.tanh(np.random.randn(2))  # random action

        # action = (steering angle, throttle amt)
        return action
