# The code is shared on SDSC Github
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# reproducible
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions, # 2 actions
            n_features, # 4 features
            learning_rate=0.01,
            reward_decay=0.95,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay # 0.95

        # initialize observation, action and reward
        self.all_observation, self.all_action, self.all_reward = [], [], []

        self.build_net() # build the network model

        # initialize a session and all variables
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def build_net(self): # build the policy function
        self.tf_obs = tf.compat.v1.placeholder(tf.float32, [None, self.n_features]) # # used as inputs to the model, 4 elements
        self.tf_acts = tf.compat.v1.placeholder(tf.int32, [None, ]) # used as labels
        self.tf_vt = tf.compat.v1.placeholder(tf.float32, [None, ])

        # layer 1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10, # 10 hidden units
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), # for weight matrix
            bias_initializer=tf.constant_initializer(0.1) # for bias
        )
        # layer 2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions, # probabilities of 2 actions
            activation=None, # will be activated later
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.all_act_prob = tf.nn.softmax(all_act)  # use softmax to convert to probabilities for 2 actions

        # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)

#        # option 1: use embedded function
#        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
#        # 'sparse_softmax_cross_entropy_with_logits' use negative log of chosen action

        # option 2: describe the formula (the idea is the same as option 1)
        neg_log_prob = tf.reduce_sum(-tf.math.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        # tf.one_hot(indices, depth) # indices: positions of 1; depth: length of each one-hot vector

        loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss (tf_vt: policy function)

        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # observation[np.newaxis, :] -> make it structure consistent with the placeholder

        # choose between actions (0 or 1) with probabilities defined by 'p'
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

        return action

    def store_transition(self, s, a, r): # store the information within the game
        self.all_observation.append(s)
        self.all_action.append(a)
        self.all_reward.append(r)

    def learn(self): # update the network
        # discount and normalize episode reward
        policy = self.policy() # used to calculate loss

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.all_observation),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.all_action),  # shape=[None, ]
             self.tf_vt: policy,  # shape=[None, ]
        })

        self.all_observation, self.all_action, self.all_reward = [], [], []    # empty episode data
        return policy

    def policy(self):
        # discount episode rewards
        policy_values = np.zeros_like(self.all_reward) # length: number of steps in each game

        running_add = 0
        # self.all_reward contains all '1's
        for t in reversed(range(0, len(self.all_reward))):
            running_add = running_add * self.gamma + self.all_reward[t] # create a decay function
            policy_values[t] = running_add

        # normalize rewards
        policy_values = policy_values - np.mean(policy_values)
        policy_values = policy_values / np.std(policy_values)
        return policy_values



