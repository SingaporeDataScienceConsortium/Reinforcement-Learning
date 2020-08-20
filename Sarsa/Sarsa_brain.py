# The code is shared on SDSC Github
import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # actions = [0,1,2,3]
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # initialize a DataFrame with column names only

    def check_state_exist(self, state):
        if state not in self.q_table.index: # if the new state is not in the Q table
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state,))
            # self.q_table.columns -> the movements [0,1,2,3] or ['up','down','left','right']
            # state -> a new state or a new coordinate

    def choose_action(self, curr_position): # self means the class itself
        self.check_state_exist(curr_position)
        # action selection
        if np.random.rand() < self.epsilon: # epsilon=0.9; 90% possibility of doing the following
            # choose best action
            state_action = self.q_table.loc[curr_position, :]

            # some actions may have the same reward. if so, randomly choose an actions
            # this may happen, especiall in early stages
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else: # 10% possibility of randomly selecting an action
            # choose random action
            action = np.random.choice(self.actions)
        return action # the return is how the explorer will move

# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):

        # use super so that SarsaTable can also inherit properties and methods from RL
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal': # if the next state is NOT terminal -> do the following calculation
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
            # for comparison, Q Learning is as below
          # q_target = r + self.gamma * self.q_table.loc[s_, :].max() # always choose the maximum

        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
