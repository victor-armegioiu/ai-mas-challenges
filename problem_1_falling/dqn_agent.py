import random
import numpy as np
from collections import deque

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras.layers import add, Input
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

import cv2 as cv2
import numpy as np

class QLayer(_Merge):
    """
        Q Layer that merges an advantage and value layer
    """
    def _merge_function(self, inputs):
        # Assumes that the inputs come in as [value, advantage]
        output = inputs[0] + (inputs[1] - K.mean(inputs[1], axis=1, keepdims=True))
        return output

"""
    Class to represent a Dueling Deep Q-Learning agent. Covers the hyper-parameters
    used in Q-Learning, such as discount rate (gamma), exploratory rate (epsilon).

    Encapsulates a medium sized neural net to approximate the Q values by performing
    experience replay. We accumulate batches of experiences and periodically train
    our model on them.
"""
class DDQNAgent(object):
    def __init__(self, state_size=(86, 86, 3), action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self._build_model()

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
        The actual underlying structure of the model. We use several convolutional layers
        in order to process the stacked frames that encapsulate our observations. 
        We use a dueling architecture where our model learns from a target model, by defining
        a value and an advantage function. 
    """
    def _build_model(self):
        model = Sequential()
        input_layer = Input(shape=self.state_size)

        conv1 = Conv2D(32, kernel_size=(8, 8),
        strides=(4, 4), 
        padding="same",
        data_format='channels_last',
        input_shape=self.state_size,
        activation='relu')(input_layer)

        conv2 = Conv2D(64, kernel_size=(4, 4),
        strides=(2, 2),
        padding="same",
        activation='relu')(conv1)
        conv3 = Conv2D(64, kernel_size=(3, 3))(conv2)

        flatten = Flatten()(conv3)
        fc1 = Dense(512)(flatten)

        advantage = Dense(self.action_size)(fc1)
        fc2 = Dense(512)(flatten)
        value = Dense(1)(fc2)
        policy = QLayer()([value, advantage]) 

        model = Model(input=[input_layer], output=[policy])
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))


        target_model = Model(input=[input_layer], output=[policy])
        target_model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))

        self.model = model
        self.target_model = target_model

    """
        This function records experiences encountered via interacting with the
        environment. We used a fixed-size deque that acts as a buffer for further
        improvements of our predictions. 
    """
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """ 
        We produce an action that matches the given state. Our exploratory strategy
        involves choosing a random action with probability epsilon (decreases with time). 
    """
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    """
        This part is the workhorse of our actual model. We sample a random batch
        of experiences from our memory buffer. We use those in order to approximate the
        quality of our actions by noting:

                    Q(s, a) = reward + gamma * max_a'(Q(s', a'))

        Given this equation, we train our network to adjust the Q values such that
        Q(s, a) converges to 'reward + gamma * max_a'(Q(s', a'))'.

        Here, the model learns from the target model's predictions of the future states.
    """
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1, 86, 86, 3))
            next_state = next_state.reshape((1, 86, 86, 3))

            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.target_model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
