import gym
from gym import spaces
import envs.construction as construction
from envs.construction_sample import sample_construction_env
from utils.construction_data import state_to_state_tensor, state_tensor_to_state
import pickle
import numpy as np
import pdb
import torch


def state2array(state, num_colored_blocks=3, goal_block_pair=None):
	'''
	Converts state object to a numpy array
	'''
	return state_to_state_tensor(state, num_colored_blocks, goal_pair=goal_block_pair).cpu().detach().numpy()


class ConstructionGymEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, initial_state, colored_block_utilities):
		super(ConstructionGymEnv, self).__init__()

		self.initial_state = initial_state
		self.num_colored_block_locations = len(self.initial_state.colored_blocks)
		self.colored_block_utilities = colored_block_utilities
		
		self.goal_block_pair = max(self.colored_block_utilities, key=self.colored_block_utilities.get)


		self.actions = list(construction.Action)
		self.action_space = spaces.Discrete(len(self.actions))

		self.observation = state2array(self.initial_state, self.num_colored_block_locations, self.goal_block_pair)

		# lower_bound = [0] * len(self.observation.shape)
		# upper_bound = list(self.observation.shape)
		# upper_bound = [x-1 for x in upper_bound]
		# print(lower_bound)
		# print(upper_bound)
		self.observation_space = spaces.Box(low=0, high=23, 
			shape=self.observation.shape, dtype=np.float64)

	def reset(self):
		'''
		For the purposes of OpenAI PPO, reset function starts environment in completely new state
		'''
		self.done = False 
		# new_env = sample_construction_env()
		# self.initial_state = new_env.initial_state
		# self.colored_block_utilities = new_env.colored_block_utilities
		self.observation = state2array(self.initial_state.clone(), self.num_colored_block_locations, self.goal_block_pair)
		return self.observation

	def step(self, action):
		converted_state = state_tensor_to_state(torch.tensor(self.observation))
		converted_action = construction.Action(action)
		state_ = self.transition(converted_state, converted_action)
		reward = self.get_reward(converted_state, converted_action)
		done = self.get_done(converted_state, converted_action)
		self.observation = state2array(state_, self.num_colored_block_locations, self.goal_block_pair)
		return self.observation, reward, done, {"state": self.observation}

	def transition(self, state, action):
		if construction.Action(action.value) == construction.Action.STOP:
			return state.clone()

		gridworld = pickle.loads(pickle.dumps(state.gridworld))
		map = gridworld.map
		block_picked = pickle.loads(pickle.dumps(state.block_picked))
		colored_blocks = pickle.loads(pickle.dumps(state.colored_blocks))

		old_pos = pickle.loads(pickle.dumps(state.agent_location))
		new_pos = construction.next_pos(old_pos, action)  # new agent location
		if construction.in_bound(new_pos, gridworld):
			tempx, tempy = new_pos
			cell = map[tempx][tempy]
			if cell == '*' or cell == '▲' or cell == "●" or \
					(cell in construction.ALL_COLORED_BLOCKS and block_picked is not None):  # conditions for not moving
				new_pos = old_pos
		x, y = new_pos  # agent location
		cell = map[x][y]

		if construction.Action(action.value) == construction.Action.PUT_DOWN and block_picked is not None and \
				cell == '.':
			block_picked = None
		elif block_picked is None and new_pos != old_pos and cell in construction.ALL_COLORED_BLOCKS:
			block_picked = cell
			construction.strModify(map, x, y, '.')
		if block_picked is not None and construction.Action(action.value) != construction.Action.PUT_DOWN:
			colored_blocks[block_picked] = new_pos
		for block, loc in colored_blocks.items():
			if loc != new_pos:  # drawing colored blocks only if an agent isn't on them
				construction.strModify(map, loc[0], loc[1], block)

		return construction.State(gridworld=construction.Gridworld(map), agent_location=new_pos, colored_blocks=colored_blocks,
		block_picked=block_picked)


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
		distance_apart = construction.manhattan_distance(loc1, loc2)
		# isDone = (distance_apart <= 1) and (construction.Action(action.value) == construction.Action.PUT_DOWN)
		# return isDone
		if block1 == next_state.block_picked or block2 == next_state.block_picked:
			return True
		return False


	def get_reward(self, state, action):
		'''
		Reward the agent for completing the task - step cost
		'''
		reward = -1.0
		if self.get_done(state, action):
			reward += 100.0
		return reward
	def close(self):
		pass



