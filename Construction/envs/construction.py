import pdb
import random
from enum import Enum
import math
import _pickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
from itertools import combinations
from copy import deepcopy

import pygame
from pygame.locals import RLEACCEL
import utils
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # initialize world with white background

ALL_COLORED_BLOCKS = ['=', 'x', '+', '!', '@', '#', '$', '%', '^', '&']
ALL_BLOCK_PAIRS = list(combinations(ALL_COLORED_BLOCKS, 2))
ALL_UTILITIES = [0 for i in range(len(ALL_BLOCK_PAIRS))]
highest_util = random.randint(0, len(ALL_BLOCK_PAIRS) - 1)
ALL_UTILITIES[0] = 100
block2color = {'=': 'yellow', 'x': 'blue', '+': 'pink', '!': 'brown', 
			'@': 'orange', '#':'indigo','$':'violet', '%': 'teal', '^': 'maroon', '&': 'lime' }
color2rgb = {'yellow': (255, 255, 0), 'blue': (0,0, 255), 'pink': (255, 192, 203), 'brown': (205, 133, 63),
		'orange': (255, 165, 0), 'indigo': (75, 0, 130), 'violet': (238, 130, 238), 'teal': (0, 128, 128),
		'maroon': (128, 0, 0), 'lime': (0, 255, 0)}

def get_box(color='black', width=20, height=20):
	surf = pygame.Surface((SCREEN_WIDTH//width, SCREEN_HEIGHT//height))

	if color in color2rgb:
		surf.fill(color2rgb[color])
	else:
		surf.fill((0, 0, 0))
	rect = surf.get_rect()
	return surf


def grid_to_pixel(x, y, width=20, height=20):
	return x * SCREEN_WIDTH // width, y * SCREEN_HEIGHT // height


class L0(pygame.sprite.Sprite):
	def __init__(self):
		super(L0, self).__init__()
		self.surf = pygame.image.load("green_agent.png").convert()  # load in L0 image
		self.surf.set_colorkey(self.surf.get_at((27, 0)), RLEACCEL)  # background is transparent
		# return a width and height of an image
		self.size = self.surf.get_size()
		# create a 0.75x bigger image than self.surf
		self.surf = pygame.transform.scale(self.surf, (int(self.size[0]*0.32), int(self.size[1]*0.32)))
		self.rect = self.surf.get_rect()


class L1(pygame.sprite.Sprite):
	def __init__(self):
		super(L1, self).__init__()
		self.surf = pygame.image.load("red_agent.png").convert()  # load in L1 image
		self.surf.set_colorkey(self.surf.get_at((0, 0)), RLEACCEL)  # background is transparent
		self.size = self.surf.get_size()
		# create a 0.75x bigger image than self.surf
		self.surf = pygame.transform.scale(self.surf, (int(self.size[0]*0.32), int(self.size[1]*0.32)))
		self.rect = self.surf.get_rect()


agent_objects = {0: L0(), 1: L1()}
color_objects = {}
for symb, col in block2color.items():
	color_objects[symb] = get_box(col)
color_objects['*'] = get_box()


def get_free_cells(cell_values):
	free_cells = []
	for row in range(len(cell_values)):
		for col in range(len(cell_values[0])):
			if cell_values[row][col] == '.':
				free_cells.append((row, col))
	return free_cells


def sample_agent_location(cell_values):
	return random.choice(get_free_cells(cell_values))


def strModify(string_array, x, y, val):
	# temp = list(string_array[x])
	# temp[y] = val
	# string_array[x] = ''.join(temp)
	string_array[x] = string_array[x][:y] + val + string_array[x][y+1:]


def manhattan_distance(loc1, loc2):
	return abs(loc2[1] - loc1[1]) + abs(loc2[0] - loc1[0])

def euclidean_distance(loc1, loc2):
	return ((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)**0.5

def md_check(loc1, loc2):
	return manhattan_distance(loc1, loc2) <= 1


def get_default_colored_block_utilities(num_possible_block_pairs):
	"""Get a default dictionary with block pairs as keys and utilities as values.

	Args
		num_possible_block_pairs (int)

	Returns (dict with string keys and int values)

		key (tuple) is the symbols of the block pair
		value (int) is the utility of that pair

		Contains utilities for all possible colored blocks in the map
		len(colored_block_utilities) can be > len(initial_state.colored_blocks)
	"""
	ALL_UTILITIES = [0 for _ in range(len(ALL_BLOCK_PAIRS))]
	highest_util = random.randint(0, num_possible_block_pairs - 1)
	ALL_UTILITIES[highest_util] = 100
	return dict(
		zip(ALL_BLOCK_PAIRS[:num_possible_block_pairs], ALL_UTILITIES[:num_possible_block_pairs], )
	)


class Cell(Enum):
	NOTHING = 0
	WALL = 1
	COLORED_BLOCK = 2
	SECOND_AGENT = 3


class Action(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	PUT_DOWN = 4
	STOP = 5


class Gridworld:
	"""Specifies location of walls and colored blocks (not what color the blocks there are though)

	Args
		cell_values: 2D list of values in {'.', '*', '=', 'x', '+'} -- see Cell
	"""

	def __init__(self, cell_values):
		self.map = cell_values
		# self.cells = []
		# for cell_values_row in cell_values:
		# 	row = []
		# 	for cell_value in cell_values_row:
		# 		if cell_value == '.':
		# 			row.append(Cell(0))
		# 		elif cell_value == '*':
		# 			row.append(Cell(1))
		# 		elif cell_value == '▲':  # second agent
		# 			row.append(Cell(3))
		# 		else:
		# 			row.append(Cell(2))
		# 	self.cells.append(row)
		self.num_rows = len(self.map)
		self.num_cols = len(self.map[0])
		# self.num_rows = len(self.cells)
		# self.num_cols = len(self.cells[0])

	def clone(self):
		return Gridworld(self.map)

	def get_cells(self, cell):
		"""Return a list of coordinates of particular cell types"""
		result = []
		for row in self.map:
			for col, val in enumerate(row):
				if val == cell:
					result.append((row, col))
		# for (row, col), val in np.ndenumerate(self.cells):
		# 	if val == cell:
		# 		result.append((row, col))
		return result

	def get_wall_locations(self):
		"""Return a list of wall coordinates"""
		return self.get_cells('*')

	def get_second_agent(self):
		return self.get_cells('▲')

	def get_colored_block_locations(self):
		"""Return a list of food truck coordinates"""
		result = []
		for row in range(self.num_rows):
			for col in range(self.num_cols):
				if self.map[row][col] in block2color:
					result.append((row, col))
		return result

	def get_cell(self, row, col):
		return self.map[row][col]

	def is_visible(self, cell_a, cell_b):
		"""Can assume full visibility for the construction example"""
		# for wall in self.get_wall_locations():
		#     if not utils.geometry.is_visible(cell_A, cell_B, wall):
		#         return False
		return True

	def __eq__(self, other):
		return (
				self.num_rows == other.num_rows
				and self.num_cols == other.num_cols
				and self.cells == other.cells
		)


class State:
	"""
	Fully visible state of the world.

	Args
		gridworld (Gridworld)
		agent_location: tuple of row, col
		colored_blocks: dictionary of colored block types to location tuple (row, col)
	"""

	def __init__(self, gridworld, agent_location, colored_blocks, block_picked=None):
		self.gridworld = gridworld
		self.agent_location = agent_location
		self.colored_blocks = colored_blocks
		self.block_picked = block_picked  # should be the key of a block picked by agent1 in colored_blocks

	def clone(self):
		return State(self.gridworld.clone(), self.agent_location, self.colored_blocks, self.block_picked)

	def __eq__(self, other):
		return (
				self.gridworld == copy.deepcopy(other.gridworld)
				and self.agent_location == other.agent_location
				and self.colored_blocks == other.colored_blocks
				and self.block_picked == other.block_picked
		)

	def __str__(self):
		gridworld = pickle.loads(pickle.dumps(self.gridworld))
		map = gridworld.map
		for k, v in self.colored_blocks.items():
			row, col = v
			strModify(map, row, col, k)
		row, col = self.agent_location
		strModify(map, row, col, "●")
		result = "\n--State--\n"
		for row in map:
			result += row + "\n"
		result += "Inventory: " + str(self.block_picked) + "\n"
		result += "Agent Location: " + str(self.agent_location) + "\n"
		result += "Block Locations: " + str(self.colored_blocks) + "\n"
		result += "--------\n"
		return result

	def pyGamePlot(self, ax=None):
		screen.fill((255, 255, 255))


		num_rows, num_cols = self.gridworld.num_rows, self.gridworld.num_cols
		#  create border of walls
		for i in range(num_rows):
			box = color_objects['*']
			screen.blit(box, grid_to_pixel(i, 0))
			screen.blit(box, grid_to_pixel(i, num_cols-1))
		for j in range(num_cols):
			box = color_objects['*']
			screen.blit(box, grid_to_pixel(0, j))
			screen.blit(box, grid_to_pixel(num_rows-1, j))

		# plot colored blocks
		for block, loc in self.colored_blocks.items():
			box = color_objects[block]
			screen.blit(box, grid_to_pixel(loc[1], loc[0]))

		# plot agent
		agent_loc = self.agent_location
		agent_id = 0

		agent_obj = agent_objects[agent_id]
		if self.block_picked is not None:
			# background = color2rgb[block2color[self.agent_inv[agent_id]]]
			background = agent_obj.surf.get_at((0, 0))
			agent_obj.surf.set_colorkey(background, RLEACCEL)
		screen.blit(agent_obj.surf, grid_to_pixel(agent_loc[1], agent_loc[0]))

		return screen

	def plot(self, ax=None):
		gridworld = self.gridworld
		map = gridworld.map
		agent_location = self.agent_location
		colored_blocks = self.colored_blocks

		num_rows = gridworld.num_rows
		num_cols = gridworld.num_cols
		if ax is None:
			_, ax = plt.subplots()

		ax.set_axis_off()
		table = matplotlib.table.Table(ax)

		width, height = 1.0 / num_cols, 1.0 / num_rows
		current_food_truck_id = 0
		for row in range(num_rows):
			for col in range(num_cols):
				fontproperties = matplotlib.font_manager.FontProperties(
					family="sans-serif", size="x-large"
				)
				val = map[row][col]
				if val == '.':
					facecolor = "white"
					text = ""
				elif val == '*':
					facecolor = "black"
					text = ""
				elif val == '▲':
					facecolor = 'red'
					text = 'Second Agent'
				else:  # colored block
					facecolor = block2color[val]
					text = ""

				if agent_location == (row, col):
					assert val != '*'
					facecolor = 'lightgray'
					text = 'Agent - Items in Inventory: '
					if self.block_picked is not None:
						text += block2color[self.block_picked] + " block"
					else:
						text += 'None'

				table.add_cell(
					row,
					col,
					width,
					height,
					text=text,
					facecolor=facecolor,
					fontproperties=fontproperties,
					loc="center",
				)
		ax.add_table(table)
		return ax

	__repr__ = __str__


class ConstructionEnv:
	"""Construction truck MDP

	Args
		initial_state (State)
		colored_block_utilities (dictionary): tuple (block1, block2) mapped to utility of being next to each other
	"""

	def __init__(self, initial_state, colored_block_utilities):
		self.initial_state = initial_state
		self.colored_block_utilities = colored_block_utilities

		self.state = self.initial_state
		self.action_space = list(Action)
		self.num_actions = len(self.action_space)

	def reset(self):
		self.state = self.initial_state.clone()
		return self.get_observation()

	def clone(self):
		env_ = ConstructionEnv(self.initial_state, self.colored_block_utilities)
		env_.state = self.state.clone()
		return env_

	def get_observation(self):
		return self.state

	def get_done(self, state, action):
		"""Is the episode done after the agent does this action?

		Args
			state (State)
			action (Action)

		Returns (bool)
		"""
		next_state = self.transition(state, action)
		# - Get colored_blocks present
		colored_blocks = next_state.colored_blocks
		# What is agent 0's most desired block pair on the map?
		# -- Block pair utilities present
		colored_block_utilities_on_the_map = {}
		for k, v in self.colored_block_utilities.items():
			b1, b2 = k
			if b1 in colored_blocks and b2 in colored_blocks:
				colored_block_utilities_on_the_map[k] = v
		# -- Take the argmax
		most_favorite_block_pair_on_the_map = max(
			colored_block_utilities_on_the_map, key=colored_block_utilities_on_the_map.get
		)
		block1, block2 = most_favorite_block_pair_on_the_map
		loc1, loc2 = next_state.colored_blocks[block1], next_state.colored_blocks[block2]
		distance_apart = manhattan_distance(loc1, loc2)
		isDone = (distance_apart <= 1) and (Action(action.value) == Action.PUT_DOWN)
		return isDone

	def get_reward(self, state, action):
		reward = -1.0
		if self.get_done(state, action):
			reward += 100.0
		return reward

	def transition(self, state, action):
		if Action(action.value) == Action.STOP:
			return state.clone()
		
		# gridworld = pickle.loads(pickle.dumps(state.gridworld))
		# gridworld = state.gridworld
		# map = pickle.loads(pickle.dumps(gridworld.map))
		# block_picked = pickle.loads(pickle.dumps(state.block_picked))
		# colored_blocks = pickle.loads(pickle.dumps(state.colored_blocks))
		
		# gridworld = deepcopy(state.gridworld)
		# map = gridworld.map
		# gridworld = state.gridworld
		# map = deepcopy(gridworld.map)
		# block_picked = deepcopy(state.block_picked)
		# colored_blocks = deepcopy(state.colored_blocks)

		state_ = pickle.loads(pickle.dumps(state))
		gridworld = state_.gridworld
		map = gridworld.map
		block_picked = state_.block_picked
		colored_blocks = state_.colored_blocks

		# old_pos = pickle.loads(pickle.dumps(state.agent_location))
		old_pos = state.agent_location
		new_pos = next_pos(old_pos, action)  # new agent location
		if in_bound(new_pos, gridworld):
			tempx, tempy = new_pos
			cell = map[tempx][tempy]
			if cell == '*' or cell == '▲' or cell == "●" or \
					(cell in ALL_COLORED_BLOCKS and block_picked is not None):  # conditions for not moving
				new_pos = old_pos
		x, y = new_pos  # agent location
		cell = map[x][y]

		if Action(action.value) == Action.PUT_DOWN and block_picked is not None and \
				cell == '.':
			block_picked = None
		elif block_picked is None and new_pos != old_pos and cell in ALL_COLORED_BLOCKS:
			block_picked = cell
			strModify(map, x, y, '.')
		if block_picked is not None and Action(action.value) != Action.PUT_DOWN:
			colored_blocks[block_picked] = new_pos
		for block, loc in colored_blocks.items():
			if loc != new_pos:  # drawing colored blocks only if an agent isn't on them
				strModify(map, loc[0], loc[1], block)

		state_.gridworld = gridworld
		state_.gridworld.map = map
		state_.colored_blocks = colored_blocks
		state_.block_picked = block_picked
		state_.agent_location = new_pos
		# state_ = State(gridworld=Gridworld(map), agent_location=new_pos, colored_blocks=colored_blocks,
		# 			 block_picked=block_picked)	

		return state_

	def step(self, action):
		state_ = self.transition(self.state, action)
		reward = self.get_reward(self.state, action)
		done = self.get_done(self.state, action)
		self.state = state_
		observation = self.get_observation()
		return observation, reward, done, {"state": self.state}

	def __str__(self):
		return (
			"----- ConstructionEnv -----"
			# "\nInitial state:"
			# f"{self.initial_state}\n"
			"Current state:"
			f"{self.state}\n"
			"Agent Inventory:\n"
			f"{self.state.block_picked}\n"
		)

	def plot(self):
		pass

	__repr__ = __str__


class StateMultiAgent:
	"""
	State of the multi-agent environment.
	Contains locations of the agents and the world state.

	Args
		gridworld (Gridworld)
		agent_locations: dictionary of tuple of row, col
		colored_blocks: dictionary of colored block types to location tuple (row, col)
	"""

	def __init__(self, gridworld, agent_locations, colored_blocks, agent_inv=None):
		self.gridworld = gridworld
		self.agent_locations = agent_locations
		self.colored_blocks = colored_blocks
		if agent_inv is None:
			self.agent_inv = {}  # initialize agents to not have any block picked
			for agents in self.agent_locations:
				self.agent_inv[agents] = None
		else:
			self.agent_inv = agent_inv

	def clone(self):
		return StateMultiAgent(self.gridworld.clone(), self.agent_locations, self.colored_blocks, self.agent_inv)

	def plot(self, ax=None):
		screen.fill((255, 255, 255))


		num_rows, num_cols = self.gridworld.num_rows, self.gridworld.num_cols
		assert num_rows == 20 
		assert num_cols == 20
		#  create border of walls
		for i in range(num_rows):
			box = color_objects['*']
			screen.blit(box, grid_to_pixel(i, 0))
			screen.blit(box, grid_to_pixel(i, num_cols-1))
		for j in range(num_cols):
			box = color_objects['*']
			screen.blit(box, grid_to_pixel(0, j))
			screen.blit(box, grid_to_pixel(num_rows-1, j))

		# plot colored blocks
		for block, loc in self.colored_blocks.items():
			box = color_objects[block]
			screen.blit(box, grid_to_pixel(loc[1], loc[0]))

		# plot agents
		for agent_id, agent_loc in self.agent_locations.items():
			agent_obj = agent_objects[agent_id]
			if self.agent_inv[agent_id] is not None:
				# background = color2rgb[block2color[self.agent_inv[agent_id]]]
				background = agent_obj.surf.get_at((0, 0))
				agent_obj.surf.set_colorkey(background, RLEACCEL)
			screen.blit(agent_obj.surf, grid_to_pixel(agent_loc[1], agent_loc[0]))

		return screen


		# assert len(self.agent_locations) == 2
		# num_rows = self.gridworld.num_rows
		# num_cols = self.gridworld.num_cols
		# if ax is None:
		#     _, ax = plt.subplots()
		#
		# ax.set_axis_off()
		# table = matplotlib.table.Table(ax)
		#
		# width, height = 1.0 / num_cols, 1.0 / num_rows
		# for row in range(len(self.gridworld.map)):
		#     for col in range(len(self.gridworld.map[0])):
		#         val = self.gridworld.map[row][col]
		#         fontproperties = matplotlib.font_manager.FontProperties(
		#             family="sans-serif", size="x-large"
		#         )
		#         if val == '.':
		#             facecolor = "white"
		#             text = ""
		#         elif val == '*':
		#             facecolor = "black"
		#             text = ""
		#         elif val in ALL_COLORED_BLOCKS:
		#             facecolor = block2color[val]
		#             text = block2color[val]
		#
		#         if self.agent_locations[0] == (row, col):
		#             assert val != '*'
		#             facecolor = 'lightblue'
		#             text = 'Agent 0'
		#         if self.agent_locations[1] == (row, col):
		#             assert val != '*'
		#             facecolor = 'red'
		#             text = 'Agent 1'
		#
		#         if self.agent_locations[0] == (row, col) and self.agent_locations[1] == (row, col):
		#             assert val != '*'
		#             facecolor = 'purple'
		#             text = 'Agents 0 & 1'
		#
		#         table.add_cell(
		#             row,
		#             col,
		#             width,
		#             height,
		#             text=text,
		#             facecolor=facecolor,
		#             fontproperties=fontproperties,
		#             loc="center",
		#         )
		# ax.add_table(table)
		# agent_L2_inv = self.agent_inv[0]
		# if agent_L2_inv is not None:
		#     agent_L2_inv = block2color[agent_L2_inv]
		# agent_L1_inv = self.agent_inv[1]
		# if agent_L1_inv is not None:
		#     agent_L1_inv = block2color[agent_L1_inv]
		# ax.set_title(f"Agent L2 Inventory: {agent_L2_inv} \n"
		#              f"Agent L1 Inventory: {agent_L1_inv}")
		# return ax

	def __eq__(self, other):
		return (
				self.gridworld == other.gridworld
				and self.agent_locations == other.agent_locations
				and self.colored_blocks == other.colored_blocks
		)

	def __str__(self):
		gridworld = pickle.loads(pickle.dumps(self.gridworld))
		map = gridworld.map

		result = "\n--State--\n"
		for k, v in self.colored_blocks.items():
			row, col = v
			strModify(map, row, col, k)

		for k, v in self.agent_locations.items():
			row, col = v
			if k == 0:
				strModify(map, row, col, "●")
			else:
				strModify(map, row, col, "▲")

		for row in map:
			result += row + "\n"
		result += "--------\n"
		return result

	__repr__ = __str__


class ConstructionMultiAgentEnv:
	"""Construction multi-agent MDP

	Args
		initial_state (StateMultiAgent)
		colored_block_utilities (dict) {0: colored_block_utilities_L0, 1: colored_block_utilities_L1}
	"""

	def __init__(
			self, initial_state, colored_block_utilities, seek_conflict=True
	):
		self.initial_state = initial_state
		self.colored_block_utilities = colored_block_utilities
		self.num_possible_block_pairs = len(self.colored_block_utilities[0])
		self.state = self.initial_state
		self.action_space = list(Action)
		self.agent_ids = list(initial_state.agent_locations.keys())
		self.num_agents = len(self.agent_ids)
		self.seek_conflict = seek_conflict

	def reset(self):
		self.state = self.initial_state
		return self.get_observation({0: None, 1: None}, {0: None, 1: None}, self.state.colored_blocks,
									self.state.agent_inv, self.state.gridworld)

	def get_observation(self, prev_agent_locations, prev_actions, prev_colored_blocks, prev_inv, prev_grid):
		"""Observations for each agent
		Only implemented for two agents at the moment.
		Assuming full observability for agents in the construction setting
		"""
		# agent_L1_loc = self.state.agent_locations[1]
		grid = pickle.loads(pickle.dumps(self.state.gridworld))
		# map = grid.map
		# for cells in grid.get_second_agent():
		#     strModify(map, cells[0], cells[1], '.')  # removing second agent duplicates
		# strModify(map, agent_L1_loc[0], agent_L1_loc[1], '▲')  # drawing second agent in as a wall

		agent_0_location = self.state.agent_locations[0]
		agent_1_location = self.state.agent_locations[1]
		observed_colored_blocks = pickle.loads(pickle.dumps(self.state.colored_blocks))
		agent_0_inv = self.state.agent_inv[0]
		agent_1_inv = self.state.agent_inv[1]
		agent_0_obs = State(grid, agent_0_location, observed_colored_blocks, agent_0_inv)

		agent_1_obs = StateL1(self.colored_block_utilities, [(agent_0_obs, 1)], agent_0_obs, agent_1_location,
							  agent_1_inv, prev_actions[0])

		return {
			"gridworld": agent_0_obs.gridworld,
			"colored_blocks": pickle.loads(pickle.dumps(self.state.colored_blocks)),
			"current_agent_locations": self.state.agent_locations,
			"prev_agent_locations": prev_agent_locations,
			"prev_actions": prev_actions,
			"prev_colored_blocks": prev_colored_blocks,
			"prev_inv": prev_inv,
			"prev_grid": prev_grid,
			"agent_0_observation": agent_0_obs,
			"agent_1_observation": agent_1_obs
		}

	def transition(self, state, actions):
		"""Transition p(next_state | state, actions)
		Moves to the desired location if it's not a wall or outside of boundaries
		Does NOT update self.state

		Args
			state (StateMultiAgent)
			actions (Dictionary of Action)

		Returns (State)
		"""
		gridworld = pickle.loads(pickle.dumps(state.gridworld))
		map = gridworld.map
		colored_blocks = pickle.loads(pickle.dumps(state.colored_blocks))
		agent_locations = pickle.loads(pickle.dumps(state.agent_locations))
		agent_inv = pickle.loads(pickle.dumps(state.agent_inv))

		walls = gridworld.get_wall_locations()
		for agent_id, agent_location in agent_locations.items():
			assert agent_location not in walls

		agent_ids = list(actions.keys())
		random.shuffle(agent_ids)
		next_agent_locations = {0: None, 1:None}
		for agent_id in agent_ids:
			action = actions[agent_id]
			old_pos = state.agent_locations[agent_id]
			new_pos = next_pos(old_pos, action)
			block_picked = agent_inv[agent_id]

			collision = False
			for agents, locs in agent_locations.items():
				if agents != agent_id:
					new_locs = next_agent_locations[agents]
					if new_pos == locs or new_pos == new_locs:
						collision = True

			if in_bound(new_pos, gridworld):
				tempx, tempy = new_pos
				cell = map[tempx][tempy]
				if cell == '*' or collision or \
						(cell in ALL_COLORED_BLOCKS and block_picked is not None):  # conditions for not moving
					new_pos = old_pos

			x, y = new_pos
			cell = map[x][y]

			if Action(action.value) == Action.PUT_DOWN and block_picked is not None and \
					cell == '.':
				strModify(map, x, y, block_picked)
				block_picked = None
			elif block_picked is None and new_pos != old_pos and cell in ALL_COLORED_BLOCKS:
				block_picked = cell
				strModify(map, x, y, '.')
			if block_picked is not None and action != Action.PUT_DOWN:
				colored_blocks[block_picked] = new_pos

			agent_inv[agent_id] = block_picked
			next_agent_locations[agent_id] = new_pos
		
		# pdb.set_trace()
		# next_state = pickle.loads(pickle.dumps(state))
		# next_state.gridworld.map = map
		# next_state.agent_locations = next_agent_locations
		# next_state.colored_blocks = colored_blocks
		# next_state.agent_inv = agent_inv
		next_state = StateMultiAgent(Gridworld(map), next_agent_locations, colored_blocks, agent_inv)
		return next_state

	def get_reward(self, state, actions):
		"""Reward R(state, action)
		For now, give zero reward unless Agent 1 brings 2 blocks together, or Agent 0 reaches target
		In that case, give Agent 0 reward
			0 (unsuccessful deception) if Agent 1 brings together the two blocks that Agent 0
				has the highest utility of
			+100 if Agent 0 creates max utility pair
			+1 (successful deception)

		Args
			state (State)
			action (Action)

		Returns (float)
		"""
		gridworld = pickle.loads(pickle.dumps(state.gridworld))
		colored_block_locations = gridworld.get_colored_block_locations()
		rewards = {0: -1.0, 1: -1.0}  # init rewards to -1 because of step cost
		if Action(actions[1].value) == Action.STOP and state.agent_locations[1] in colored_block_locations:
			# Has agent 0 deceived agent 1?
			# - Get colored_blocks present
			colored_blocks = state.colored_blocks
			# What is agent 0's most desired block pair on the map?
			# -- Block pair utilities present
			colored_block_utilities_on_the_map = {}
			for k, v in self.colored_block_utilities[0].items():
				b1, b2 = k
				if b1 in colored_blocks and b2 in colored_blocks:
					colored_block_utilities_on_the_map[k] = v

			# -- Take the argmax
			most_favorite_block_pair_on_the_map = set(max(
				colored_block_utilities_on_the_map, key=colored_block_utilities_on_the_map.get
			))

			# What is/are the colored block pair/pairs agent 1 created?
			pairs_made = []
			for neighbor in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
				next_loc = next_pos(state.agent_locations[1], neighbor)
				if in_bound(next_loc, gridworld) and next_loc not in gridworld.get_wall_locations() \
						and next_loc != state.agent_locations[0]:
					pairs_made.append({gridworld.map[state.agent_locations[1][0]][state.agent_locations[1][1]],
									   gridworld.map[next_loc[0]][next_loc[1]]})

			deceived = most_favorite_block_pair_on_the_map not in pairs_made
			if deceived and self.seek_conflict:
				rewards[0] += 1
			elif not deceived and not self.seek_conflict:
				rewards[0] += 1
			elif deceived and not self.seek_conflict:
				rewards[0] -= 1
			else:
				rewards[0] -= 1

		# Reward agent 0 if they successfully reach the target
		if self.get_done(state, actions):
			rewards[0] += 100  # for completing the goal
			if self.seek_conflict:
				rewards[1] -= 100
			else:
				rewards[1] += 100
		return rewards

	def get_done(self, state, actions):
		"""Is the episode done?"""
		# -- Block pair utilities present
		colored_blocks = state.colored_blocks
		colored_block_utilities_on_the_map = {}
		for k, v in self.colored_block_utilities[0].items():
			b1, b2 = k
			if b1 in colored_blocks and b2 in colored_blocks:
				colored_block_utilities_on_the_map[k] = v

		# -- Take the argmax
		most_favorite_block_pair_on_the_map = max(
			colored_block_utilities_on_the_map, key=colored_block_utilities_on_the_map.get
		)

		# Done if both blocks are 1 unit away
		block1, block2 = most_favorite_block_pair_on_the_map
		loc1, loc2 = colored_blocks[block1], colored_blocks[block2]
		distance_apart = manhattan_distance(loc1, loc2)
		distance_check = distance_apart <= 1
		return distance_check

	def step(self, actions):
		"""Gym-style step function"""
		rewards = self.get_reward(self.state, actions)
		done = self.get_done(self.state, actions)
		prev_agent_locations = pickle.loads(pickle.dumps(self.state.agent_locations))
		prev_colored_blocks = pickle.loads(pickle.dumps(self.state.colored_blocks))
		prev_inv = pickle.loads(pickle.dumps(self.state.agent_inv))
		prev_grid = pickle.loads(pickle.dumps(self.state.gridworld))
		self.state = self.transition(self.state, actions)
		observations = self.get_observation(prev_agent_locations, actions, prev_colored_blocks, prev_inv, prev_grid)
		return observations, rewards, done, {"state": self.state}


class StateL1:
	"""State of the L1 environment
	Lightweight container of L0's utilities/desires, belief and observation.
	"""

	def __init__(self, colored_block_utilities_L0, belief_L0, observation_L0, agent_location_L1, agent_inv_L1, prev_action_0=None):
		self.colored_block_utilities_L0 = colored_block_utilities_L0
		self.belief_L0 = belief_L0
		self.observation_L0 = observation_L0
		self.agent_location_L1 = agent_location_L1
		self.agent_inv_L1 = agent_inv_L1
		self.prev_action_L0 = prev_action_0


class ObservationL1:
	"""Observation of the L1 environment as described by
	Lightweight container of L0's state and action.
	"""

	def __init__(self, state_L0, action_L0):
		self.state_L0 = state_L0
		self.action_L0 = action_L0

	def __str__(self):
		return (
			"\n---- L1 observation ----\n"
			"L0 state: "
			f"{self.state_L0}\n"
			"L0 action: "
			f"{self.action_L0}\n"
			"------------------------\n"
		)

	__repr__ = __str__


class ConstructionEnvL1:
	"""A construction environment for the all-knowing L1 agent which fully observes an L0 agent.
	L1 tries to harm L0 by creating one of the block pairs L0 does not desire
	(regardless of whether it is in the world)

	Args
		seek_conflict (bool)
		base_colored_block_utilities_L1 (dict)
		initial_state_L0 (State)
		colored_block_utilities_L0 (dictionary)
			key (tuple of (str, str)) is a block pair
			value (int) is the utility of that block pair

			Contains utilities for all possible block pairs in the map
			len(colored_block_utilities_L0) can be > len(initial_state_L0.colored_blocks)
	"""

	def __init__(
			self,
			seek_conflict,
			base_colored_block_utilities_L1,
			initial_state_L0,
			colored_block_utilities_L0,
			agent_location_L1,
			agent_inv_L1,
			beta_L0=0.01,
			beta_L1=0.01
	):
		self.seek_conflict = seek_conflict
		self.base_colored_block_utilities_L1 = base_colored_block_utilities_L1
		self.initial_state_L0 = initial_state_L0
		self.colored_block_utilities_L0 = colored_block_utilities_L0
		self.beta_L0 = beta_L0
		self.beta_L1 = beta_L1

		self.num_possible_block_pairs = len(self.colored_block_utilities_L0)
		self.gridworld = self.initial_state_L0.gridworld
		self.colored_blocks = self.initial_state_L0.colored_blocks
		self.action_space = list(Action)
		self.timestep = 0
		self.initial_agent_location_L1 = agent_location_L1
		self.initial_agent_inv_L1 = agent_inv_L1
		self.agent_inv_L1 = agent_inv_L1
		self.agent_location_L1 = self.reset_L1_loc()

		# Sample rollout
		self.env_L0 = ConstructionEnv(self.initial_state_L0, self.colored_block_utilities_L0)
		# self.rollout_L0, _ = sample_L0_rollout(self.env_L0, self.beta_L0)
		# self.num_timesteps = len(self.rollout_L0)
		# self.actions_L0, self.states_L0, self.obss_L0, self.rewards_L0 = list(zip(*self.rollout_L0))

	def reset_L1_loc(self):
		return self.initial_agent_location_L1

	def reset(self):
		self.timestep = 0
		self.agent_location_L1 = self.reset_L1_loc()
		self.agent_inv_L1 = self.initial_agent_inv_L1

		return ObservationL1(self.initial_state_L0, None)

	def get_observation(self):
		"""Observation of the L1 agent

		Returns
			obs (ObservationL1) which consists of state_L0 and action_L0
		"""
		if self.timestep < self.num_timesteps:
			return ObservationL1(self.states_L0[self.timestep], self.actions_L0[self.timestep])
		else:
			pdb.set_trace()
			return None

	def transition(self, agent_location_L1, action):
		"""Transition p(next_state | state, action)
		Moves to the desired location with a block if it's not a wall or outside of boundaries
		Does NOT update self.state
		This is deterministic at the moment.

		Args
			agent_location_L1: tuple of row, col
			action (Action)

		Returns next_agent_location_L1 (tuple)
		"""

		if Action(action.value) == Action.STOP:
			return agent_location_L1, self.agent_inv_L1, self.gridworld
		gridworld = pickle.loads(pickle.dumps(self.gridworld))
		map = gridworld.map
		agent_inv_L1 = self.agent_inv_L1
		state_L0 = self.get_observation().state_L0
		loc_L0 = state_L0.agent_location
		colored_blocks = state_L0.colored_blocks

		old_pos = agent_location_L1
		new_pos = next_pos(old_pos, action)  # new agent location
		if in_bound(new_pos, gridworld):
			tempx, tempy = new_pos
			cell = map[tempx][tempy]
			if cell == '*' or loc_L0 == new_pos or \
					(cell in ALL_COLORED_BLOCKS and agent_inv_L1 is not None):  # conditions for not moving
				new_pos = old_pos

		x, y = new_pos  # agent location
		cell = map[x][y]

		if Action(action.value) == Action.PUT_DOWN and agent_inv_L1 is not None and \
				cell == '.':
			strModify(map, x, y, agent_inv_L1)
			agent_inv_L1 = None
		elif agent_inv_L1 is None and new_pos != old_pos and cell in ALL_COLORED_BLOCKS:
			agent_inv_L1 = cell
			strModify(map, x, y, '.')
		if agent_inv_L1 is not None and action != Action.PUT_DOWN:
			colored_blocks[agent_inv_L1] = new_pos
		gridworld = Gridworld(map)

		return new_pos, agent_inv_L1, gridworld

	def get_reward(self, agent_location_L1, action):
		"""Reward for the L1 agent
		-1 for making a move, something else for creating a pair of blocks next to each other

		Args
			state (State)
			action (Action)

		Returns (float)
		"""
		gridworld = self.gridworld
		colored_block_locations = gridworld.get_colored_block_locations()
		reward = -1.0
		if Action(action.value) == Action.STOP and agent_location_L1 in colored_block_locations:
			# Has agent 0 deceived agent 1?
			# - Get colored_blocks present
			colored_blocks = self.colored_blocks
			# What is agent 0's most desired block pair on the map?
			# -- Block pair utilities present
			colored_block_utilities_on_the_map = {}
			for k, v in self.colored_block_utilities_L0.items():
				b1, b2 = k
				if b1 in colored_blocks and b2 in colored_blocks:
					colored_block_utilities_on_the_map[k] = v

			# -- Take the argmax
			most_favorite_block_pair_on_the_map = set(max(
				colored_block_utilities_on_the_map, key=colored_block_utilities_on_the_map.get
			))

			# What is/are the colored block pair/pairs agent 1 created?
			pairs_made = []
			for neighbor in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
				next_loc = next_pos(agent_location_L1, neighbor)
				if in_bound(next_loc, gridworld) and next_loc not in gridworld.get_wall_locations \
						and gridworld.map[next_loc[0]][next_loc[1]] in block2color:
					pairs_made.append({gridworld.map[agent_location_L1[0]][agent_location_L1[1]],
									   gridworld.map[next_loc[0]][next_loc[1]]})

			for pair in pairs_made:
				if pair != most_favorite_block_pair_on_the_map and self.seek_conflict:  # incentivize hurting L0
					reward += 100
				elif pair == most_favorite_block_pair_on_the_map and not self.seek_conflict:
					reward += 100
				elif pair != most_favorite_block_pair_on_the_map and not self.seek_conflict:
					reward -= 100
				else:
					reward -= 100
		return reward

	def get_done(self, agent_location_L1, action):
		"""Is the episode done after the agent does this action?

		Args
			agent_location_L1
			action (Action)

		Returns (bool)
		"""
		gridworld = self.gridworld

		colored_block_locs = gridworld.get_colored_block_locations()
		for loc1 in colored_block_locs:
			for loc2 in colored_block_locs:
				if loc1 != loc2 and md_check(loc1, loc2) and Action(action.value) == Action.PUT_DOWN:
					return True

		return self.timestep >= self.num_timesteps - 1

	def step(self, action):
		"""Gym-style step function"""
		reward = self.get_reward(self.agent_location_L1, action)
		done = self.get_done(self.agent_location_L1, action)
		self.agent_location_L1, self.agent_inv_L1, self.gridworld = self.transition(self.agent_location_L1, action)
		self.timestep += 1
		observation = self.get_observation()
		return observation, reward, done, {"agent_location_L1": self.agent_location_L1,
										   "agent_inventory_L1": self.agent_inv_L1}

	def __str__(self):
		return (
			"\n---- ConstructionEnvL1 ----"
			"\nInitial L0 state:"
			f"{self.initial_state_L0}\n"
			"Current L1 observation:"
			f"{self.get_observation()}\n"
			"L0's Block pair utilities:\n"
			f"{self.colored_block_utilities_L0}\n"
			f"L1 agent location: {self.agent_location_L1}\n"
			"------------------------\n"
		)

	__repr__ = __str__


def in_bound(location, gridworld):
	x, y = location
	return (0 <= x < gridworld.num_rows) and (0 <= y < gridworld.num_cols)


def next_pos(location, action):
	"""What's the location after taking an action `action`
	Args
		location (tuple of row, col)
		action (Action)

	Returns tuple of row, col
	"""

	assert len(location) == 2
	action = Action(action.value)
	if action == Action.UP:
		shift = [-1, 0]
	elif action == Action.DOWN:
		shift = [1, 0]
	elif action == Action.LEFT:
		shift = [0, -1]
	elif action == Action.RIGHT:
		shift = [0, 1]
	else:
		shift = [0, 0]
	return (location[0] + shift[0], location[1] + shift[1])
	# return tuple(np.array(location) + shift)


def get_state_L0_with_agent_location_L1_str(state, agent_location_L1):
	gridworld = state.gridworld
	map = gridworld.map

	result = "\n--State--\n"
	for k, v in state.colored_blocks.items():
		row, col = v
		temp = list(map[row])
		temp[col] = k
		map[row] = "".join(temp)

	row, col = state.agent_location
	temp = list(map[row])
	temp[col] = "●"
	map[row] = "".join(temp)

	row, col = agent_location_L1
	temp = list(map[row])
	temp[col] = "▲"
	map[row] = "".join(temp)

	for row in map:
		result += row + "\n"
	result += "--------\n"
	return result


def colored_block_utilities_clash(colored_block_utilities_L0, colored_block_utilities_L1):
	"""Do the colored_block_pair utilities clash?

	Args
		colored_block_utilities_L0 (dict of str: float)
		colored_block_utilities_L1 (dict of str: float)

	Returns bool
	"""
	# Get the key of colored_block_utilities_L0 which has the max value
	max_utility_colored_block_L0 = max(colored_block_utilities_L0, key=colored_block_utilities_L0.get)
	# Do the same for colored_block_utilities_L1
	max_utility_colored_block_L1 = max(colored_block_utilities_L1, key=colored_block_utilities_L1.get)
	return max_utility_colored_block_L0 == max_utility_colored_block_L1


def get_colored_block_utilities_L1(
		colored_block_utilities_L0, base_colored_block_utilities_L1, seek_conflict, state
):
	"""Compute colored block utilities of an L1 agent based on its social goal (seek_conflict)
	and L0's colored block utilities.

	Args
		colored_block_utilities_L0 (dict with string keys and int values)
		base_colored_block_utilities_L1 (dict with string keys and int values)
		seek_conflict (bool)

	Returns (dict with string keys and int values)
	"""
	if not seek_conflict:
		return pickle.loads(pickle.dumps(colored_block_utilities_L0))
	else:
		# The result
		colored_block_utilities_L1 = pickle.loads(pickle.dumps(base_colored_block_utilities_L1))

		# Get the key of colored_block_utilities_L0 which has the max value
		max_utility_colored_block_L0 = max(colored_block_utilities_L0, key=colored_block_utilities_L0.get)
		# Do the same for colored_block_utilities_L1
		max_utility_colored_block_L1 = max(colored_block_utilities_L1, key=colored_block_utilities_L1.get)
		goal_blocks = list(max_utility_colored_block_L1)
		if max_utility_colored_block_L0 == max_utility_colored_block_L1:
			new_L1_util_dict = {}
			for (block1, block2), v in colored_block_utilities_L1.items():  # only including pairs where both blocks on map
				x1, y1 = state.colored_blocks[block1]
				cell_1 = state.gridworld.map[x1][y1]
				x2, y2 = state.colored_blocks[block2]
				cell_2 = state.gridworld.map[x2][y2]
				if cell_1 == block1 and cell_2 == block2 and (block1, block2) != max_utility_colored_block_L1 and \
						state.block_picked is None:
					new_L1_util_dict[(block1, block2)] = v
				else:
					if (state.block_picked == block1 and block2 not in goal_blocks) or \
							(state.block_picked == block2 and block1 not in goal_blocks):
						new_L1_util_dict[(block1, block2)] = v

			second_largest_utility_colored_block_L1 = max(
				new_L1_util_dict,
				key=new_L1_util_dict.get,
			)
			# Then, swap the two largest values
			colored_block_utilities_L1[max_utility_colored_block_L1] = base_colored_block_utilities_L1[
				second_largest_utility_colored_block_L1
			]
			colored_block_utilities_L1[
				second_largest_utility_colored_block_L1
			] = base_colored_block_utilities_L1[max_utility_colored_block_L1]

		return colored_block_utilities_L1


def get_num_rankings(num_colored_block_locations):
	"""How many different block_pairs are there given the number of colored_blocks."""
	return int(math.factorial(num_colored_block_locations) / (math.factorial(2) * math.factorial(num_colored_block_locations-2)))
