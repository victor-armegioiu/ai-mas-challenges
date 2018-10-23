import random
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

import cv2 as cv2

"""
    Class to represent a Deep Q-Learning agent. Covers the hyper-parameters
    used in Q-Learning, such as discount rate (gamma), exploratory rate (epsilon).

    Encapsulates a medium sized neural net to approximate the Q values by performing
    experience replay. We accumulate batches of experiences and periodically train
    our model on them.
"""
class DQNAgent(object):
    def __init__(self, state_size=(86, 86, 1), action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    """
        Huber loss : https://en.wikipedia.org/wiki/Huber_loss
    """
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    """
        The actual underlying structure of the model. We mostly focus on the image
        representation part by using two waves of CONV=>MAX_POOLING layers. 

        We ultimately flatten the results and proceed by moving into smaller and 
        smaller densely connected latent spaces.
    """
    def _build_model(self):
    	model = Sequential()
    	model.add(Conv2D(20, (5, 5), padding="same", data_format='channels_last', 
    			input_shape=self.state_size, activation='linear'))
    	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    	model.add(Conv2D(50, (5, 5), padding="same", activation='linear'))
    	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    	model.add(Flatten())
    	model.add(Dense(512, activation='relu'))
    	model.add(Dense(64, activation='relu'))
    	model.add(Dense(self.action_size, activation='linear'))
    	model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
    	return model

    """
        This function records experiences encountered via interacting with the
        environment. We used a fixed-size deque that acts as a buffer for further
        improvements of our predictions. 

        We preprocess the image in order to make it grayscale, 
        hence it becomes easier to process by our model, while minimizing the relevant
        information that is lost.
    """
    def remember(self, state, action, reward, next_state, done):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY).reshape((1, 86, 86, 1))
        next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY).reshape((1, 86, 86, 1))
        self.memory.append((state, action, reward, next_state, done))

    """ 
        We produce an action that matches the given state. Our exploratory strategy
        involves choosing a random action with probability epsilon (decreases with time). 
    """
    def act(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY).reshape((1, 86, 86, 1))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    """
        This part is the workhorse of our actual model. We sample a random batch
        of experiences from our memory buffer. We use those in order to approximate thq
        quality of our actions by noting:

                    Q(s, a) = reward + gamma * max_a'(Q(s', a'))

        Given this equation, we train our network to adjust the Q values such that
        Q(s, a) converges to 'reward + gamma * max_a'(Q(s', a'))'.
    """
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
