# The code is shared on SDSC Github
"""
source: https://gym.openai.com/envs/CartPole-v0/
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The system is controlled by applying a force of +1 or -1 to the cart.
The pendulum starts upright, and the goal is to prevent it from falling over.
A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 12 degrees from vertical, or the cart moves more than 2.4 units from the center.
"""
import gym
from PG_brain import PolicyGradient
import tensorflow as tf
import matplotlib.pyplot as plt # used for commented commands later
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

env = gym.make('CartPole-v0') # initialize the game environment
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped # extract the raw environment
# you can uncomment the following two lines to check the first two argument to DeepQNetwork
#print(env.action_space.n) # 2 actions: left and right (the maze example has 4 actions)
#print(env.observation_space.shape[0]) # 4 features (the maze example has 2 features, x and y coordinates)

# observation is used to describe a state in an environment
# each observation contains 4 items
# x: cart position along the horizontal axis, between -2.4 and 2.4
# x_dot: speed
# theta: angle between the pole and the vertical axis
# theta_dot: angular speed

RL = PolicyGradient(
    n_actions=env.action_space.n, # two actions, left and right
    n_features=env.observation_space.shape[0], # 4 features
    learning_rate=0.02,
    reward_decay=0.99,
)

for game in range(300):

    curr_observation = env.reset()
    # initial observations may be slightly different after reset()

    while True:
        # fresh env
        env.render()

        # RL choose action based on curr_observation
        action = RL.choose_action(curr_observation)

        # RL take action and get next curr_observation and reward
        next_observation, reward, done, info = env.step(action)

        # store the information into memory
        RL.store_transition(curr_observation, action, reward)

        if done:
            game_rewards = sum(RL.all_reward) # calculate total reward of the current game

            print("Game:", game, " reward:", int(game_rewards))

            vt = RL.learn() # perform learning when the game is completed

#            # uncomment the following to check vt
#            print(vt)
#            plt.plot(vt)    # plot the episode vt
#            plt.xlabel('Steps within each game')
#            plt.ylabel('normalized policy value')
#            plt.show()
#            fig = plt.gcf()
#            fig.canvas.draw()

            break

        curr_observation = next_observation

env.close()