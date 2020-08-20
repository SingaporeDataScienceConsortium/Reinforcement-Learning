# The code is shared on SDSC Github
import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from matplotlib import pyplot as plt
import matplotlib
from chainer import serializers
plt.close()

data_raw = pd.read_csv('Data/stock.txt') # load data
data = pd.read_csv('Data/stock.txt') # load data
data['Date'] = pd.to_datetime(data['Date']) #convert Dates to 'datetime' type
data = data.set_index('Date') #set Date as the index of the dataframe
print('Earliest timestamp:',data.index.min())
print('Latest   timestamp:',data.index.max())
print(data.index.min(), data.index.max())
data.head() # get the first 5 (default) records

date_split = '2016-01-04'
train = data[:date_split]
test = data[date_split:]
train_raw = data_raw[:data_raw['Date'].tolist().index('2016-01-04')]
test_raw = data_raw[data_raw['Date'].tolist().index('2016-01-04'):]
print('Length of training data:',train.shape[0])
print('Length of test data:    ',test.shape[0])


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


# ------------------------------------ load -----------------------------------
Q = Q_Network(input_size=90+1, hidden_size=100, output_size=3)
serializers.load_npz('model/DQN.model', Q)
#serializers.load_npz('model_fromInstructor1/DQN.model', Q)
#serializers.load_npz('model_fromInstructor2/DQN.model', Q)



# train
train_env = Environment1(train)
pobs = train_env.reset()
train_acts = []
train_rewards = []
train_profit = []
for _ in range(len(train_env.data)-1):
    pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
    pact = np.argmax(pact.data)
    train_acts.append(pact)
    obs, reward, done = train_env.step(pact)
    train_rewards.append(reward)
    train_profit.append(train_env.profits)
    pobs = obs
train_profits = train_env.profits


# test
test_env = Environment1(test)
pobs = test_env.reset()
test_acts = []
test_rewards = []
test_profit = []
for _ in range(len(test_env.data)-1):
    pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
    pact = np.argmax(pact.data)
    test_acts.append(pact)
    obs, reward, done = test_env.step(pact)
    test_rewards.append(reward)
    test_profit.append(test_env.profits)
    pobs = obs
test_profits = test_env.profits


# display results for training data
fig, (ax1,ax2) = plt.subplots(2,1, sharey=False, sharex=True)
dates_train = list(matplotlib.dates.datestr2num(train_raw['Date']))
c_train = train['Close'].tolist()
close = [[c_train[i] * 1] for i in range(len(dates_train))]
ax1.plot(dates_train,close)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
ax1.grid(True)
ax1.set_title('Stock and Actions (Training Data)')
buy_label = 1
sell_label = 1
for i in range(len(train_acts)):
    if train_acts[i]==1 and buy_label==1:
        ax1.scatter(dates_train[i],close[i],c='g',alpha=0.5,label='buy')
        buy_label = 0
    if train_acts[i]==1:
        ax1.scatter(dates_train[i],close[i],c='g',alpha=0.5)
    if train_acts[i]==2 and sell_label==1:
        ax1.scatter(dates_train[i],close[i],c='r',alpha=0.5,label='sell')
        sell_label = 0
    if train_acts[i]==2:
        ax1.scatter(dates_train[i],close[i],c='r',alpha=0.5)
ax1.legend(loc='upper left')
ax2.plot(dates_train,train_profit)
ax2.grid(True)
ax2.set_title('Profit (Training Data)')


# display results for test data
fig, (ax1,ax2) = plt.subplots(2,1, sharey=False, sharex=True)
dates_test = list(matplotlib.dates.datestr2num(test_raw['Date']))
c_test = test['Close'].tolist()
close = [[c_test[i] * 1] for i in range(len(dates_test))]
ax1.plot(dates_test,close)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
ax1.grid(True)
ax1.set_title('Stock and Actions (Test Data)')
buy_label = 1
sell_label = 1
for i in range(len(test_acts)):
    if test_acts[i]==1 and buy_label==1:
        ax1.scatter(dates_test[i],close[i],c='g',alpha=0.5,label='buy')
        buy_label = 0
    if test_acts[i]==1:
        ax1.scatter(dates_test[i],close[i],c='g',alpha=0.5)
    if test_acts[i]==2 and sell_label==1:
        ax1.scatter(dates_test[i],close[i],c='r',alpha=0.5,label='sell')
        sell_label = 0
    if test_acts[i]==2:
        ax1.scatter(dates_test[i],close[i],c='r',alpha=0.5)
ax1.legend(loc='upper left')
ax2.plot(dates_test[1:],test_profit)
ax2.grid(True)
ax2.set_title('Profit (Test Data)')



