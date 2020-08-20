# The code is shared on SDSC Github
import numpy as np
import pandas as pd
import time
import copy
import chainer # for deep net construction
import chainer.functions as F
import chainer.links as L
from matplotlib import pyplot as plt
import matplotlib
import os
from chainer import serializers


data_raw = pd.read_csv('Data/stock.txt') # load data
data = pd.read_csv('Data/stock.txt') # load data
data['Date'] = pd.to_datetime(data['Date']) #convert Dates to 'datetime' type
data = data.set_index('Date') #set Date as the index of the dataframe
print('Earliest timestamp:',data.index.min())
print('Latest   timestamp:',data.index.max())
print(data.index.min(), data.index.max())
data.head() # get the first 5 (default) records

# create a folder to save the trained model
if not os.path.exists('model/'):
    os.makedirs('model/')

date_split = '2016-01-04'
train = data[:date_split]
test = data[date_split:]
train_raw = data_raw[:data_raw['Date'].tolist().index('2016-01-04')]
test_raw = data_raw[data_raw['Date'].tolist().index('2016-01-04'):]
print('Length of training data:',train.shape[0])
print('Length of test data:    ',test.shape[0])

fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)

dates_train = list(matplotlib.dates.datestr2num(train_raw['Date']))
c_train = train['Close'].tolist()
fig.autofmt_xdate()
fig.tight_layout()

ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
ax1.set_title('Train')

close = [[c_train[i] * 1] for i in range(len(dates_train))]
data_train = pd.DataFrame(close, index=dates_train, columns=["close"])
data_train["close"].plot(ax=ax1)
ax1.grid(True)
#plt.show()

dates_test = list(matplotlib.dates.datestr2num(test_raw['Date']))
c_test = test['Close'].tolist()

ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
ax2.set_title('Test')
close = [[c_test[i] * 1] for i in range(len(dates_test))]
data_test = pd.DataFrame(close, index=dates_test, columns=["close"])
data_test["close"].plot(ax=ax2)
fig.autofmt_xdate()
fig.tight_layout()
ax2.grid(True)
#plt.show()


class Environment1:

    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = 0
        self.done = False # all through the code, done is always False; is never changed to True
        self.profits = 0
        self.positions = [] # to store buy prices
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)] # a list (history_t=90) of '0's
        return [self.position_value] + self.history # obs; a list (90+1) of '0's

    def step(self, act):
        reward = 0

        # act = 0: stay,      1: buy,      2: sell
        if act == 1: # buy; assumption: if buy, only buy one stock
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif act == 2: # sell; assumption: if sell, sell all stocks
            if len(self.positions) == 0: # to be trained to do not sell when you have bought nothing
                reward = -1 # if there is no BUY record yet, give a small penalty
            else:
                profits = 0

                # check how much we have earned so far; check through history
                for p in self.positions:
                    profits = profits + (self.data.iloc[self.t, :]['Close'] - p)

                reward = reward + profits # this is the reward for this step; closely related to profit

                self.profits = self.profits + profits # total profit since last reset
                self.positions = [] # remove BUY records if you sell

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions: # if not sell yet, check accumulated profit
            self.position_value = self.position_value + (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0) # remove the first element
        self.history.append(np.round(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'],4)) # record profit between two consecutive time points

        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        self.position_value = np.round(self.position_value,2)

        return [self.position_value] + self.history,   reward,    self.done # obs, reward, done
        # [self.position_value]: the first element is always the accumulated profit


env = Environment1(train)


class Q_Network(chainer.Chain): # Chain is used to write neural net

    def __init__(self, input_size, hidden_size, output_size):
        super(Q_Network, self).__init__(
            fc1 = L.Linear(input_size, hidden_size),
            fc2 = L.Linear(hidden_size, hidden_size),
            fc3 = L.Linear(hidden_size, output_size)
        )

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h) # note activated
        return y

    def reset(self):
        self.zerograds() # reset all gradients to 0


# ------------------------------------ train ----------------------------------
Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3) # output has 3 elements -> 3 actions
Q_ast = copy.deepcopy(Q)
optimizer = chainer.optimizers.Adam()
optimizer.setup(Q)

epoch_num = 50
step_max = len(env.data)-1 # train within the data size
memory_size = 200
batch_size = 20
epsilon = 1.0
epsilon_decrease = 1e-3 # 0.001
epsilon_min = 0.1
start_reduce_epsilon = 200
train_freq = 10
update_q_freq = 20
gamma = 0.97
show_log_freq = 5

memory = []
total_step = 0
total_rewards = []
total_losses = []

start = time.time()
print('epoch\t\tepsilon\t\ttotal_step\tlog_reward\tlog_loss\telapse_time')
for epoch in range(epoch_num):

    pobs = env.reset() # reset the environment for each epoch; pobs is a list of position values
    step = 0
    done = False
    total_reward = 0
    total_loss = 0

    while not done and step < step_max: # train within the data size

        # select act
        pact = np.random.randint(3) # choose an action
        if np.random.rand() > epsilon:
            # most of the time; epsilon will decrease later
            # at the beginning, the engine should follow the Q table more, as time goes on, it can try some randomness
            pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1)) # pobs is a list of position values; previous obs
            pact = np.argmax(pact.data) # choose the max action

        # collect results from the selected action
        obs, reward, done = env.step(pact)

        # add memory
        memory.append((pobs, pact, reward, obs, done))
        if len(memory) > memory_size:
            memory.pop(0) # remove the oldest record

        # train or update q
        if len(memory) == memory_size: # when we have enough memory
            if total_step % train_freq == 0: # execute training
                shuffled_memory = np.random.permutation(memory) # shuffle
                memory_idx = range(len(shuffled_memory))
                for i in memory_idx[::batch_size]: # choose data every 'batch_size' records
                    batch = np.array(shuffled_memory[i:i+batch_size]) # extract one batch

                    # seperate 5 elements in memory
                    b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                    b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                    b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                    q = Q(b_pobs) # pass info to the the Q net
                    maxq = np.max(Q_ast(b_obs).data, axis=1) # prediction for multiple data
                    target = copy.deepcopy(q.data)
                    for j in range(batch_size):
                        target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                    Q.reset()
#                    print(q,'next',target)
                    loss = F.mean_squared_error(q, target)
                    total_loss += loss.data
                    loss.backward()
                    optimizer.update()

            if total_step % update_q_freq == 0: # update parameters in the net
                Q_ast = copy.deepcopy(Q)

        # epsilon
        if epsilon > epsilon_min and total_step > start_reduce_epsilon:
            epsilon -= epsilon_decrease

        # next step
        total_reward += reward
        pobs = obs
        step += 1
        total_step += 1

    total_rewards.append(total_reward)
    total_losses.append(total_loss)

    if (epoch+1) % show_log_freq == 0:
        log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
        log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
        elapsed_time = time.time()-start
        print('\t\t'.join(map(str, [epoch+1, np.round(epsilon,2), np.round(total_step,2), np.round(log_reward,2), np.round(log_loss,2), np.round(elapsed_time,2)])))
        start = time.time()

serializers.save_npz('model/DQN.model', Q)
print('model saved')
# ------------------------------------ train ----------------------------------


## --------------------------- commands for load ------------------------------
#Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
#serializers.load_npz('model/DQN.model', Q)
## --------------------------- commands for load ------------------------------

# display loss
fig, (ax1,ax2) = plt.subplots(1,2, sharey=False, sharex=False)
ax1.set_title('loss')
total_losses_df = pd.DataFrame(total_losses, index=np.arange(len(total_losses)), columns=["total_losses"])
total_losses_df["total_losses"].plot(ax=ax1)
ax1.grid(True)
ax1.set_xlabel('epoch')
plt.show()

# display rewards
ax2.set_title('reward')
total_rewards_df = pd.DataFrame(total_rewards, index=np.arange(len(total_rewards)), columns=["total_rewards"])
total_rewards_df["total_rewards"].plot(ax=ax2)
ax2.grid(True)
ax2.set_xlabel('epoch')
plt.show()

