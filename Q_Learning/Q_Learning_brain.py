# The code is shared on SDSC Github
import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # actions = [0,1,2,3]
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # initialize a DataFrame with column names only

    def choose_action(self, curr_position): # self means the class itself
        self.check_state_exist(curr_position)
        # action selection
        if np.random.uniform() < self.epsilon: # epsilon=0.9; 90% possibility of doing the following
            # choose the best action
            state_action = self.q_table.loc[curr_position, :]

            # some actions may have the same reward. if so, randomly choose an actions
            # this may happen, especiall in early stages
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else: # 10% possibility of randomly selecting an action
            # choose random action
            action = np.random.choice(self.actions)
        return action # the return is how the explorer will move

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a] # get the value of the previous state
        if s_ != 'terminal': # if the next state is NOT terminal -> do the following calculation
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] = self.q_table.loc[s, a] + self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):

        if state not in self.q_table.index: # if the new state is not in the Q table

            # append a new state to the Q table
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state,))
            # self.q_table.columns -> the movements [0,1,2,3] or ['up','down','left','right']
            # state -> a new state or a new coordinate