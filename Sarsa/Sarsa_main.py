# The code is shared on SDSC Github
from Sarsa_env import Maze
from Sarsa_brain import SarsaTable


def update():
    for game in range(100): # 100 rounds
        # initial curr_position
        curr_position = env.reset()
        # the return is a list of coordinates of the top-left and bottom-right positions
        # [x1, y1, x2, x2]
        # since it's the return of reset(), the return is always [5.0, 5.0, 35.0, 35.0]

        # RL choose action based on curr_position
        action = RL.choose_action(str(curr_position))

        step_counter = 0
        while True:
            # fresh env
            env.render()

            # RL take action and get next position and reward
            next_position, reward, done = env.step(action)

            # RL choose action based on next_position
            next_action = RL.choose_action(str(next_position))

            # learn from this movement
            RL.learn(str(curr_position), action, reward, str(next_position), next_action)

            # swap position and action
            curr_position = next_position
            action = next_action

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

    # Sarsa table initialization
    RL = SarsaTable(actions=list(range(env.n_actions))) # input: [0,1,2,3] -> a list

    update()