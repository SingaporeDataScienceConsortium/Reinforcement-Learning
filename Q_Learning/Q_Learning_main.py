# The code is shared on SDSC Github
# Q learning: Q table
# example: Maze

from Q_Learning_env import Maze
from Q_Learning_brain import QLearningTable

def update():
    for game in range(100): # 100 rounds

        # initial position
        curr_position = env.reset()
        # the return is a list of coordinates of the top-left and bottom-right positions
        # [x1, y1, x2, x2]
        # since it's the return of reset(), the return is always [5.0, 5.0, 35.0, 35.0]

        step_counter = 0
        while True:
            env.render()

            # RL choose action based on position
            action = RL.choose_action(str(curr_position))

            # RL take action and get next position and reward
            next_position, reward, done = env.step(action)

            # learn from this movement
            RL.learn(str(curr_position), action, reward, str(next_position))

            # swap positions
            curr_position = next_position

            # fresh env
            env.render()

            step_counter = step_counter + 1

            # break while loop when end of this game
            if done=='treasure' or done=='trap':
                break
        print('==================================================================')
        print('\nGame',game+1,'.',step_counter, 'steps used to',done,'.')
#        print(RL.q_table) # optional print the latest Q table

    # end of game
    print('Game Completed.')
    env.destroy() # use tkinter.destroy() to close the interface

if __name__ == "__main__": # execute the following if this script is run directly
    env = Maze() # maze (class) initialization

    # Q table initialization
    RL = QLearningTable(actions=list(range(env.n_actions))) # input: [0,1,2,3] -> a list

    update()
