# The code is shared on SDSC Github
from DQN_env import Maze
from DQN_brain import DeepQNetwork


def run_maze():
    total_step = 1
    for game in range(300): # 100 rounds
        # initial curr_position
        curr_position = env.reset()
        # the return is a list of coordinates of the top-left and bottom-right positions
        # [x1, y1, x2, x2]
        # since it's the return of reset(), the return is always [5.0, 5.0, 35.0, 35.0]
        local_step = 1
        while True:
            # fresh env
            env.render()

            # RL choose action based on curr_position
            action = RL.choose_action(curr_position)

            # RL take action and get next curr_position and reward
            next_position, reward, done = env.step(action)


            # store the information into memory
            RL.store_transition(curr_position, action, reward, next_position)

            # perform learning when there are enough records (200 records) in memory
            # perform learning every 5 steps
            if (total_step > 200) and (total_step % 5 == 0):
                RL.learn()


            # swap curr_position
            curr_position = next_position

            env.render()

            # break while loop when end of this episode
            if done=='treasure' or done=='trap':
                break
            total_step += 1
            local_step += 1
#        print('==================================================================')
        print('Game',game+1,'.',local_step, 'steps used to',done,'.','Global step =',total_step,'.')

    # end of game
    print('Game Completed.')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,learning_rate=0.01,reward_decay=0.9,
                      e_greedy=0.9,replace_target_net=200,memory_size=2000,)
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()