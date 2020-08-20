# The code is shared on SDSC Github
# simple reinforcement learning
# example: 1D command line

import numpy as np
import pandas as pd
import time

N_STATES = 6   # the size of the 1D world.6 positions including the treasure
ACTIONS = ['left', 'right']     # 2 actions

EPSILON = 0.9   # define greedy police
# 90% of the time: follow the Q table
# 10% of the time: randomly choose between 'left' and 'right'

# for updating Q table
ALPHA = 0.1    # learning rate
GAMMA = 0.9    # discount factor

MAX_RoundS = 15   # maximum Rounds
FRESH_TIME = 0.3  # pause time for one move


# use pandas to build a table
def build_q_table(n_states, actions): # build_q_table(6, ['left','right'])
    table = pd.DataFrame(np.zeros((n_states, len(actions))),  # q_table initialization with 0s
                         columns=actions)                     # actions's name: 'left' and 'right'
    return table


def choose_action(state, q_table): # this function is used to choose an action
    state_actions = q_table.iloc[state, :] # retrieve the 'left' and 'right' values of the specified row
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # execute the following for two conditions:
        # (1) 10% probability (sampling from uniform distribution)
        # (2) the corresponding state have all '0's for both 'left' and 'right'
        action_name = np.random.choice(ACTIONS) # randomly choose and action: 'left' or 'right'
    else:
        # execute the following for all other conditions
        action_name = state_actions.idxmax()    # choose the action with larger value
    return action_name
    # the return is the action, 'left' or 'right'


def get_env_feedback(S, A): # 2 inputs: (1) state (2) name of action
    # this function is used to interact with the environment get the next state and reward

    if A == 'right':    # if move to the right
        if S == N_STATES - 2:   # terminate
            # this is the situation where its right is the treasure (terminal); the current postion is the 5th
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0 # if its right is not the terminal, there is no reward
    else:   # if move to the left
        R = 0 # if move to the left, the reward is always 0
        if S == 0: # if reach the wall
            S_ = S  # go back to its current position
        else:
            S_ = S - 1
    return S_, R
    # two returns: (1) next state (2) reward


def update_env(S, Round, step_counter):
    # this function is used to update the environment
    # initial: '-----T'
    env_list = ['-']*(N_STATES-1) + ['T']   # initialize the environment
    if S == 'terminal': # if the treasure is found
        print('\rRound',Round+1,'total steps =',step_counter,'.')
        # '\r' means moving the cursor to the beginning of the line (where the printing starts)
        time.sleep(2)
    else: # if the treasure is not found
        env_list[S] = 'o' # move the searcher to a particular position by change '-' to 'o'

        interaction = ''.join(env_list)
        # convert list to string with insertion of '' (in this example, it's empty) between every 2 elements

        # display option 1
#        print(interaction,' step',step_counter) # to display all steps

        # display option 2
        print('\r',interaction,' step',step_counter, end='') # dynamic display
        # ' end='' ' means clearing the current line

        time.sleep(FRESH_TIME)


q_table = build_q_table(N_STATES, ACTIONS) # return a data frame
for Round in range(MAX_RoundS):

    # settings before each round
    step_counter = 0
    S = 0
    is_terminated = False

    # initialize a Q table with all '0's
    update_env(S, Round, step_counter)

    while not is_terminated:
        # is_terminated = True:  break
        # is_terminated = False: continue

        A = choose_action(S, q_table) # A is 'left' or 'right'
        q_predict = q_table.loc[S, A] # get the current Q value
        S_, R = get_env_feedback(S, A)  # collect 2 outputs: (1) next state (2) reward

        if S_ != 'terminal': # if the next state is not 'terminal', perform the following calculation
            q_target = R + GAMMA * q_table.iloc[S_, :].max()
            # q_table.iloc[S_, :].max(): maximum action probability in the next state
        else: # if the next state is 'terminal', directly assign the full reward
            q_target = R
            is_terminated = True    # terminate this Round

        q_table.loc[S, A] =q_table.loc[S, A] + ALPHA * (q_target - q_predict)  # update the Q table
        S = S_  # move to next state

        update_env(S, Round, step_counter+1) # update the environmen; update the position of searcher
        step_counter += 1

#    print(q_table)

# print the final Q table
print('\r\nQ-table:\n')
print(q_table)
