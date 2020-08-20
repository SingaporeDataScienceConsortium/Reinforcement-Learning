# The code is shared on SDSC Github
import numpy as np
import time
import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 4  # 4 rows
MAZE_W = 4  # 4 columns

class Maze(tk.Tk): # tk.Tk defines the main interface and all main activities will be there
    def __init__(self): # define the properties of an object constructed by this class
        super(Maze, self).__init__() # use super so that Maze can also inherit properties and methods from tkinter
        self.action_space = ['u', 'd', 'l', 'r'] # 4 actions in total
        self.n_actions = len(self.action_space)
        self.title('maze') # title of the interface
        self.Build_maze() # invoke a function to build the maze

    def Build_maze(self): # self means the class itself
        # use canvas to specify the size and background color of the interface
        self.canvas = tk.Canvas(self, bg='white',height=MAZE_H * UNIT,width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT): # range(0, 4*40, 40)
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1) # draw lines

        for r in range(0, MAZE_H * UNIT, UNIT): # range(0, 4*40, 40)
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1) # draw lines

        # starting position
        origin = np.array([20, 20])

        # hell 1 (row 3 and column 2)
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(hell1_center[0]-15,hell1_center[1]-15,hell1_center[0]+15,hell1_center[1]+15,fill='black')

        # hell 2 (row 2 and column 3)
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(hell2_center[0]-15,hell2_center[1]-15,hell2_center[0]+15,hell2_center[1]+15,fill='black')

        # create treasure (row 3 and column 3)
        treasure_center = origin + UNIT * 2
        self.treasure = self.canvas.create_oval(treasure_center[0]-15,treasure_center[1]-15,treasure_center[0]+15,treasure_center[1]+15,fill='yellow')

        # create red rectangle: the explorer
        self.explorer = self.canvas.create_rectangle(origin[0]-15,origin[1]-15,origin[0]+15,origin[1]+15,fill='red')

        # pack all
        self.canvas.pack() # use pack() to organize widgets in blocks

    def reset(self):
        self.update()

        time.sleep(0.1)

        # delete the current explorer
        self.canvas.delete(self.explorer)

        origin = np.array([20, 20])

        # initialize a new explorer
        self.explorer = self.canvas.create_rectangle(origin[0]-15,origin[1]-15,origin[0]+15,origin[1]+15,fill='red')

        return self.canvas.coords(self.explorer) # return the coordinates of the reset explorer

    def step(self, action):
        s = self.canvas.coords(self.explorer)
        # s[0]: control horizontal movement
        # s[1]: control vertical movement
        base_action = np.array([0, 0])
        # noted that hitting the wall is also considered a step
        if action == 0:   # up
            if s[1] > UNIT: # if the explorer is not in row 1
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT: # if the explorer is not in the last row
                base_action[1] += UNIT
        elif action == 2:   # left
            if s[0] < (MAZE_W - 1) * UNIT: # if the explorer is not in the last column
                base_action[0] += UNIT
        elif action == 3:   # right
            if s[0] > UNIT: # if the explorer is not in column 1
                base_action[0] -= UNIT

        # the 2nd and 3rd arguments shows the movements
        self.canvas.move(self.explorer, base_action[0], base_action[1])  # move agent

        # get the coordinates of the of moved explorer
        s_ = self.canvas.coords(self.explorer)  # next state

        # reward function
        if s_ == self.canvas.coords(self.treasure): # if the next state is the treasure
            reward = 1
            done = 'treasure'
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]: # if the next state is one of the hells
            reward = -1
            done = 'trap'
            s_ = 'terminal'
        else:
            reward = 0 # for other positions, there are no rewards
            done = 'continue'

        return s_, reward, done

    def render(self):
        time.sleep(0.1)

        # complete any pending geometry management and redraw widgets as necessary
        self.update() # use tkinter.update() to update the interface


