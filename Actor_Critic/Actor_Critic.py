# The code is shared on SDSC Github
import numpy as np
import tensorflow as tf
import gym
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

np.random.seed(2)
tf.compat.v1.set_random_seed(2)

# Superparameters
MAX_GAME = 1000
GAMMA = 0.9     # reward decay
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_features = env.observation_space.shape[0] # 4 features
N_actions = env.action_space.n # 2 actions

class Actor(object):
    def __init__(self, sess, N_features, N_actions, lr=0.001):
        self.sess = sess

        self.s = tf.compat.v1.placeholder(tf.float32, [1, N_features]) # fixed size here
        self.a = tf.compat.v1.placeholder(tf.int32, None)
        self.td_error = tf.compat.v1.placeholder(tf.float32, None)  # TD_error (TD: temporal difference). This is the reward for actor

        l1_Actor = tf.layers.dense(
            inputs=self.s, # one input: state
            units=20,    # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1), # weights
            bias_initializer=tf.constant_initializer(0.1) # biases
        )

        self.acts_prob = tf.layers.dense(
            inputs=l1_Actor,
            units=N_actions,    # output units
            activation=tf.nn.softmax,   # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1), # weights
            bias_initializer=tf.constant_initializer(0.1)  # biases
        )

        log_prob = tf.math.log(self.acts_prob[0, self.a]) # convert to log probability

        self.exp_v = tf.reduce_mean(log_prob * self.td_error)
        # advantage (TD_error) guided loss. TD_error will bring extra value to the selected action
        # the extra value can be positive or negative

        self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict = {self.s: s, self.a: a, self.td_error: td})
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return an action following probabilities defined by 'p'


class Critic(object):
    def __init__(self, sess, N_features, lr=0.01):
        self.sess = sess

        self.s = tf.compat.v1.placeholder(tf.float32, [1, N_features])
        self.v_ = tf.compat.v1.placeholder(tf.float32, [1, 1])
        self.r = tf.compat.v1.placeholder(tf.float32, None)

        l1_Critic = tf.layers.dense(
            inputs=self.s,
            units=20,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1), # weights
            bias_initializer=tf.constant_initializer(0.1)  # biases
        )

        self.v = tf.layers.dense(
            inputs=l1_Critic,
            units=1,  # output units
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., .1), # weights
            bias_initializer=tf.constant_initializer(0.1)  # biases
        )

        self.td_error = self.r + GAMMA * self.v_ - self.v # this equation follows Q learning method
        self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.compat.v1.Session()
actor = Actor(sess, N_features=N_features, N_actions=N_actions, lr=LR_A)
critic = Critic(sess, N_features=N_features, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
sess.run(tf.compat.v1.global_variables_initializer())

for game in range(MAX_GAME):
    s = env.reset()
    step = 0
    track_r = []
    while True:
        env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_ # move to the next state
        step += 1

        if done:
            game_reward = sum(track_r)
            print("Game:", game, ". Reward:", int(game_reward),'.',step,'steps.')
            break

env.close()