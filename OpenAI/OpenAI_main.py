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
from OpenAI_brain import DeepQNetwork

env = gym.make('CartPole-v0') # initialize the game environment

env = env.unwrapped # extract the raw environment

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01,
                  e_greedy=0.9,
                  replace_target_net=100,
                  memory_size=2000,
                  e_greedy_increment=0.001,)

# you can uncomment the following two lines to check the first two argument to DeepQNetwork
#print(env.action_space.n) # 2 actions: left and right (the maze example has 4 actions)
#print(env.observation_space.shape[0]) # 4 features (the maze example has 2 features, x and y coordinates)

# observation is used to describe a state in an environment
# each observation contains 4 items
# x: cart position along the horizontal axis, between -2.4 and 2.4
# x_dot: speed
# theta: angle between the pole and the vertical axis
# theta_dot: angular speed

total_steps = 1

for game in range(100):

    curr_position = env.reset()
    game_reward = 0
    while True:
        # fresh env
        env.render()

        # RL choose action based on curr_position
        action = RL.choose_action(curr_position)

        # RL take action and get next curr_position and reward
        next_position, reward, done, info = env.step(action)
        # info: used for debugging but not applicable in this example

        x, x_dot, theta, theta_dot = next_position
        # x: cart position along the horizontal axis, between -2.4 and 2.4
        # x_dot: speed
        # theta: angle between the pole and the vertical axis
        # theta_dot: angular speed

        r1 = (env.x_threshold - abs(x))/env.x_threshold
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians
        # env.x_threshold = 2.4
        # env.theta_threshold_radians = 0.2 (12 degrees)
        reward = r1 + r2
        # the reward considers both the position of the cart and the angle of the pole
        # the cart should be close to the center
        # the pole should be upright

        # store the information into memory
        RL.store_transition(curr_position, action, reward, next_position)

        game_reward += reward
        if total_steps > 1000 : # the performance will be better if the learning frequency is higher
            RL.learn()

        if done:
            print('Game:', game,'. Reward in this game:', round(game_reward, 2),' epsilon:', round(RL.epsilon, 2),'.')
            # RL.epsilon: probability of choosing larger Q value
            break

        curr_position = next_position
        total_steps += 1

env.close()

