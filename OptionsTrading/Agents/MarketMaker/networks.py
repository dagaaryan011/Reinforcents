import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam


class selector(keras.Model):
    def __init__(self):
        super(selector, self).__init__()
        
        self.indiv1 = Dense(16, activation=None)
        self.leaky1 = LeakyReLU(alpha=0.01)

        self.indiv2 = Dense(8, activation=None)
        self.leaky2 = LeakyReLU(alpha=0.01) 

        self.indiv3= Dense(4, activation=None)
        self.leaky3 = LeakyReLU(alpha=0.01) 

        self.indiv4 = Dense(2, activation=None)
        self.leaky4 = LeakyReLU(alpha=0.01) 

        self.indiv5 = Dense(1, activation=None)
        self.leaky5 = LeakyReLU(alpha=0.01) 

        self.choose1 = Dense(24, activation=None)
        self.leaky10 = LeakyReLU(alpha=0.01) 

        self.choose2 = Dense(24, activation=None)
        self.leaky11 = LeakyReLU(alpha=0.01) 

        self.choose3 = Dense(24, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, state1, state2, state3, state4, state5, 
               state6, state7, state8, state9, state10, 
               state11, state12, state13, state14, state15, 
               state16, state17, state18, state19, state20,
               state21, state22, state23, state24):

        # Apply individual transformations to each state
        states = []
        for state in [state1, state2, state3, state4, state5,
                    state6, state7, state8, state9, state10,
                    state11, state12, state13, state14, state15,
                    state16, state17, state18, state19, state20,
                    state21, state22, state23, state24]:
            state = self.leaky1(self.indiv1(state))
            state = self.leaky2(self.indiv2(state))
            state = self.leaky3(self.indiv3(state))
            state = self.leaky4(self.indiv4(state))
            state = self.leaky5(self.indiv5(state))
            states.append(state)

        # Concatenate all transformed states along the last dimension
        state = tf.concat(states, axis=-1)

        # Further processing
        state = self.leaky10(self.choose1(state))
        state = self.leaky11(self.choose2(state))
        state = self.choose3(state)

        return state



class CriticNetwork(keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(32, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        self.q = Dense(1, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, state, action):
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        return self.q(x)

class ValueNetwork(keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = Dense(32, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        self.v = Dense(1, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.v(x)

class ActorNetwork(keras.Model):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.n_actions = 4
        self.noise = 1e-6
        self.fc1 = Dense(32, activation='relu')
        self.fc2 = Dense(16, activation='relu')
        self.fc3 = Dense(8, activation='relu')
        self.alpha = Dense(self.n_actions, activation='sigmoid')
        self.beta = Dense(self.n_actions, activation='sigmoid')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        alpha = self.fc3(x)
        beta = self.fc3(x)
        alpha = self.alpha(x)
        beta = self.beta(x)
        return alpha, beta

    def sample_normal(self, state, reparameterize=True):
        alpha, beta = self.call(state)
        alpha = tf.math.softplus(alpha) + 1e-6
        beta = tf.math.softplus(beta) + 1e-6
        probabilities = tfp.distributions.Beta(alpha, beta)
        action = probabilities.sample()
        log_probs = probabilities.log_prob(action)
        log_probs = tf.math.reduce_logsumexp(log_probs, axis=1, keepdims=True)
        return action, log_probs

