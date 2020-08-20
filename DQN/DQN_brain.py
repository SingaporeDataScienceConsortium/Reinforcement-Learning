# The code is shared on SDSC Github
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            # set default parameters
            n_actions, # 4 actions
            n_features, # # number of features: 2 coordinates (x and y)
            learning_rate=0.01,
            reward_decay=0.9, # used in Q value calculation

            # 90% possibility of using Q_eval
            # 10% possibility of following randomness
            e_greedy=0.9,

            replace_target_net=300, # update target net every 300 learning steps
            memory_size=500, # max number of records in memory
            batch_size=32, # a batch of x and y coordinates

            # to gradually increase the possibility of using Q_eval
            # try more random walks at the beginning
            e_greedy_increment=None,
    ):
        # copy the default parameters
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_net = replace_target_net
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment

        # e_greedy_increment = True:   do not fix the max probability of using Q_eval
        # e_greedy_increment = False:  fix the max probability of using Q_eval
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step = 0

        # initialize zero memory
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2)) # size = [500, 2*2+2]
        # s, a, r, s_ to be stored
        # s:  2 elements
        # a:  1 elements
        # r:  1 elements
        # s_: 2 elements

        # consist of [target_net, evaluate_net]
        self.build_net()
        t_params = tf.compat.v1.get_collection('target_net_params') # use the specified key to get related parameters
        e_params = tf.compat.v1.get_collection('eval_net_params') # use the specified key to get related parameters
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)] # assign e to t

        # initialize a session and all variables
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.cost_history = [] # to record all costs

    def build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input: a list of x and y coordinates
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # used to calculate loss later

        # a list of related grapp collection keys
        # used later to update target net
        c_names = ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES] # a list of related grapp collection keys

        n_l1 = 10 # 10 hidden units in the 1st hidden layer
        w_initializer = tf.random_normal_initializer(0., 0.3) # specify mean and standard deviation
        b_initializer = tf.constant_initializer(0.1) # constant

        # 1st layer
        # use tf.compat.v1.get_variable to get an existing variable
        w1_eval = tf.compat.v1.get_variable('w1_eval', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
        b1_eval = tf.compat.v1.get_variable('b1_eval', [1, n_l1], initializer=b_initializer, collections=c_names)
        l1_eval = tf.nn.relu(tf.matmul(self.s, w1_eval) + b1_eval)

        # 2nd layer
        w2_eval = tf.compat.v1.get_variable('w2_eval', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
        b2_eval = tf.compat.v1.get_variable('b2_eval', [1, self.n_actions], initializer=b_initializer, collections=c_names)
        self.q_eval = tf.matmul(l1_eval, w2_eval) + b2_eval
        # linear activation is applied above to approximate real states (Q values)

        self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        # ------------------ build target_net ------------------
        # same structure as evaluate_net
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')    # input

        # a list of related grapp collection keys
        # used later to update target net
        c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

        # 1st layer
        w1_target = tf.compat.v1.get_variable('w1_target', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
        b1_target = tf.compat.v1.get_variable('b1_target', [1, n_l1], initializer=b_initializer, collections=c_names)
        l1_target = tf.nn.relu(tf.matmul(self.s_, w1_target) + b1_target)

        # 2nd layer
        w2_target = tf.compat.v1.get_variable('w2_target', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
        b2_target = tf.compat.v1.get_variable('b2_target', [1, self.n_actions], initializer=b_initializer, collections=c_names)
        self.q_next = tf.matmul(l1_target, w2_target) + b2_target
        # linear activation is applied above to approximate real states (Q values)

    def store_transition(self, s, a, r, s_):
        # store the following
        # 1. current position
        # 2. action
        # 3. reward
        # 4. next position

        if not hasattr(self, 'memory_counter'): # check whether the self object has an attribute, 'memory_counter'
            self.memory_counter = 0

        transition = np.hstack((s, a, r, s_)) # a vector of 6 elements

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition # replace the 'index'th row

        self.memory_counter += 1

    def choose_action(self, curr_position):
        # to have the same structure of placeholder
        curr_position = curr_position[np.newaxis, :]

        if np.random.uniform() < self.epsilon: # following Q learning
            # feed the current position to the net and get Q value for all actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: curr_position})
            action = np.argmax(actions_value) # choose the largest Q values
        else: # randon walks
            action = np.random.randint(0, self.n_actions) # randomly choose an action
        return action

    def learn(self):
        # check to replace target parameters (every 300 learn_steps)
        if self.learn_step % self.replace_target_net == 0:
            self.sess.run(self.replace_target_op)
            print('\nTarget network updated.\n')

        # sample batch memory from all memory
        # if greater than 500 -> there are 500 records available
        if self.memory_counter > self.memory_size: # if greater than 500
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        # if smaller than 500 -> not enough records
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :] # extract the records

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # feed s_ -> next position
                self.s: batch_memory[:, :self.n_features],  # feed s -> current position
            })

        # change q_target based on q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32) # get batch index
        eval_act_index = batch_memory[:, self.n_features].astype(int) # a -> index of an action
        reward = batch_memory[:, self.n_features + 1] # r -> reward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) # only update selected actions

        """
        For example in this batch I have 2 samples and 4 actions:
        q_eval =
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

        q_target = q_eval =
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

        Then change q_target with the real q_target value based on the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is 2;
            sample 1, I took action 2, and the max q_target value is 2:
        q_target =
        [[2, 3, 1, 2],
         [4, 5, 2, 1]]

        So the (q_target - q_eval) becomes:
        [[(2)-(1), 0, 0, 0],
         [0, 0, (2)-(7), 0]]

        We then backpropagate this error based on the corresponding action to network,
        leave other actions as error=0 cause we didn't choose it.
        To make other actions as error=0, the above calculation assign all q_eval to q_target first and then only update the chosen action.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
        self.cost_history.append(self.cost)

        # increasing epsilon if do not fix the max probability of using Q_eval
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        if (self.learn_step+1)%10==0:
            print('         ','Learning step =',self.learn_step+1,'.')
        self.learn_step += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



