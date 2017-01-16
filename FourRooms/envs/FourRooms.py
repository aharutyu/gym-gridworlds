import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

""" Four rooms. The goal is either in the 3rd room, or in a hallway adjacent to it
"""

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class FourRooms(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self)

    self.room_sizes = [[5,5], [6,5], [5,5], [4,5]]
    self.pre_hallways = 
                    [ 
                      [[[2,4], RIGHT, 0], [[4,1], DOWN, 3]], 
                      [[[2,0], LEFT, 0], [[5,2], DOWN, 1]],
                      [[[0,2], UP, 1], [[2,0], LEFT, 2]],
                      [[[3,4], RIGHT, 2], [[0,1], UP, 3]],
                    ]
    self.hallway_coords = [ [2,5], [6,2], [2,-1], [-1,1] ]
    self.hallways = [ #self.hallways[i][j] = [next_room, next_coord] when taking action j from hallway i#
                      [ [0, self.hallway_coords[0]], [1, [2,0]], [0, self.hallway_coords[0]], [0, [2,4]] ],
                      [ [1, [5,2]], [1, self.hallway_coords[1]], [2, [0,2]], [1, self.hallway_coords[1]] ],
                      [ [2, self.hallway_coords[2]], [2, [2,0]], [2, self.hallway_coords[2]], [3, [3,4]] ],
                      [ [0, [4,1]], [3, self.hallway_coords[3]], [3, [0,1]], [3, self.hallway_coords[3]] ]

                      # [[LEFT, 0, [2,4]], [RIGHT, 1, [2,0]]],
                      # [[UP, 1, [5,2]], [DOWN, 2, [0,2]]],
                      # [[LEFT, 3, [3,4]], [RIGHT, 2, [2,0]]],
                      # [[UP, 0, [4,1]], [DOWN, 3, [0,1]]],
                    ]
    self.noise = 0.33
    self.goal = [2, [1, 2]]

#    self.n_states = sum([nn[0] * nn[1] for nn in self.room_sizes]) + 4 + 1 # hallways and absorbing state

    # offsets[1:5] - 1 are also hallway locations
    self.offsets = [0] * 5
    for i in range(len(self.room_sizes)):
      self.offsets[i + 1] = self.offsets[i] + self.room_sizes[i][0] * self.room_sizes[i][1] + 1
    self.n_states = self.offsets[5] + 1
    self.absorbing_state = self.n_states

    self.step_reward = 0.0
    self.terminal_reward = 1.0
    self.bump_reward = -0.1

    self.start_state = 0
    self.in_hallway = False
    self.done = False
    self._reset()

    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Discrete(self.n_states) # with absorbing state

  def encode(self, room, coord):
    # TODO: absorbing state?
    #hallway = [True for ind, c in enumerate(coord) if c - self.room_sizes[room][ind] == 0]
    if self.in_hallway:
      return self.offsets[room + 1] - 1
      # maybe have hallways as input
    ind_in_room = self.coord2ind(coord, sizes=room_sizes[room])
    return ind_in_room + self.offsets[room]

  def decode(self, index):
    # if index == self.absorbing_state:
    #   return [0,0,0]
    # we don't treat absorbing state, it's instead treated in step
    #hallway = True if index - (offsets[room + 1] - 1) == 0 else False

    room = [r for r, offset in enumerate(offsets) if index < offsets[r]][0]
    if self.in_hallway:
      coord_in_room = self.hallway_coords[room]
    else:
      coord_in_room = self.ind2coord(index - offsets[room], room_sizes[room])
    return room, coord_in_room # hallway

  def _step(self, action):
    assert self.action_space.contains(action)

    if self.state == self.terminal_state:
      self.state = self.absorbing_state
      self.done = True
      return self.state, self._get_reward(), self.done, None

    [room, coord]= self.decode(self.state)

    if np.random.rand() < self.noise:
      action = self.action_space.sample()

    if self.in_hallway # hallway action
      [room2, coord2] = self.hallways[self.hallway][action]
      if coord2 not in self.hallway_coords:
        self.in_hallway = False
    elif coord in h for h in self.pre_hallways[room]: # action into a hallway
      if action == h[1]:
        self.in_hallway = True
        room2 = h[2]
        self.coord2 = self.hallway_coords[room2]
    else: # normal action
      [row, col] = coord
      [rows, cols] = room_sizes[room]
      room2 = room
      if action == UP:
        row2 = max(row - 1, 0)
      elif action == DOWN:
        row2 = min(row + 1, rows - 1)
      elif action == RIGHT:
        col2 = min(col + 1, cols - 1)
      elif action == LEFT:
        col2 = max(col - 1, 0)
      coord2 = [row2, col2]

    new_state = self.encode(room2, coord2)
    reward = self._get_reward(new_state=new_state)
    self.state = new_state

    return self.state, reward, self.done, None

  def _get_reward(self, new_state=None):
    if self.done:
      return self.terminal_reward

    reward = self.step_reward
    
    if self.bump_reward != 0 and self.state == new_state:
      reward = self.bump_reward

    return reward

  def at_border(self):
    [row, col] = self.ind2coord(self.state)
    return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)

  def ind2coord(self, index, sizes=None):
    if sizes is None:
      sizes = [self.n]*2
      
    [rows, cols] = sizes

    assert(index >= 0)
    #assert(index < self.n_states - 1)

    row = index // cols
    col = index % cols

    return [row, col]


  def coord2ind(self, coord, sizes=None):
    if sizes is None:
      sizes = [self.n]*2

    [rows, cols] = sizes
    [row, col] = coord
    assert(row < rows)
    assert(col < cols)

    return row * cols + col


  def _reset(self):
    self.state = self.start_state if not isinstance(self.start_state, str) else np.random.randint(self.n_states - 1)
    self.done = False
    self.in_hallway = False
    return self.state

  def _render(self, mode='human', close=False):
    pass
      
