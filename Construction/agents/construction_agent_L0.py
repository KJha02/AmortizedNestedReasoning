import copy
import pdb
from collections import deque
import itertools
import numpy as np
import random
import envs.construction as construction
import scipy.special
import _pickle as pickle
from enum import Enum
import concurrent.futures
import multiprocessing
import sys
import uuid
import time

def globalize(func):
	def result(*args, **kwargs):
		return func(*args, **kwargs)
	result.__name__ = result.__qualname__ = uuid.uuid4().hex
	setattr(sys.modules[result.__module__], result.__name__, result)
	return result

class Action(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	PUT_DOWN = 4
	STOP = 5

def manhattan_distance(loc1, loc2):
	return abs(loc2[0] - loc1[0]) + abs(loc2[1] - loc1[1])

def md_check(loc1, loc2):
	return manhattan_distance(loc1, loc2) <= 1

def get_match(colored_blocks, colored_blocks_observation):
	"""Flag whether a colored block matches an observed colored blocks.
	Assuming full observability in the construction case

	Args:
		colored_blocks: list/tuple of colored block names (strings of length one that are not "?")
		colored_blocks_observation: list/tuple of colored block observations
			(strings of length one, where "?" denote all possible food trucks)

	Returns (bool)
	"""
	return True



'''
Used for A* search
'''
class Node():
	def __init__(self, location, block_picked, state=None, parent=None, parentAction=None, ):
		self.parent = parent
		self.parentAction = parentAction
		self.location = location 
		self.block_picked = block_picked
		self.state = state

		self.g = 0  # distance from start
		self.h = 0  # heuristic distance to target
		self.f = 0  # joint value

	def __eq__(self, other):
		return self.location == other.location and self.block_picked == other.block_picked

	def __lt__(self, other):
		return self.f < other.f

	def __hash__(self):
		return hash((self.location, self.block_picked))


'''
Returns the length of the shortest path as calculated by A star
'''
import heapq, math, sys
def astar(state, transition, goal_loc, goal_block=None):
	if state.agent_location == goal_loc and state.block_picked is not None:
		return []

	if goal_block is None:
		goal_block = state.gridworld.map[goal_loc[0]][goal_loc[1]]

	if goal_block == '.':
		goal_block = None

	# Create start and end node
	start_node = Node(state.agent_location, state.block_picked, state, None, None, )
	start_node.g = start_node.h = start_node.f = 0
	end_node = Node(goal_loc, goal_block, None, None, None, )
	end_node.g = end_node.h = end_node.f = 0


	# Initialize both open and closed list
	open_set = set() 
	closed_set = set()  

	# Add the start node
	open_set.add(start_node)

	# Loop until you find the end
	while len(open_set) > 0:			 
		# Get the current node
		current_node = sorted(open_set, key=lambda inst:inst.f)[0]


		# Found the goal
		if current_node == end_node:
			path = []
			current = current_node
			while current is not None:
				path.append(current.parentAction)
				current = current.parent
			return path[::-1] # Return reversed path


		open_set.remove(current_node)


		if current_node in closed_set:
			continue
		closed_set.add(current_node)

		# # current_node = open_list[0]
		# current_node = heapq.heappop(open_list)
		# if current_node in open_set:
		# 	open_set.remove(current_node)

		# closed_list.append(current_node)
		# closed_set.add(current_node)

		# generate children
		action_space = list(construction.Action)
		children = []
		for action in action_space:
			# Get the state & location of the neighbor
			next_state = transition(current_node.state, action)
			next_loc = next_state.agent_location
			inv = next_state.block_picked

			newNode = Node(next_loc, inv, next_state, current_node, action)
			children.append(newNode)

		# Loop through children
		for child in children:
			# Create the f, g, and h values
			child.g = current_node.g + 1
			child.h = manhattan_distance(child.location, end_node.location)
			child.f = child.g + child.h

			add_to_open = child not in closed_set and \
				not any(open_node.f <= child.f for open_node in open_set if open_node == child)
		
			if add_to_open:
				open_set.add(child)


def bfs(state, transition, goal_loc, goal_block=None):
	if state.agent_location == goal_loc and state.block_picked is not None:
		return []

	if goal_block is None:
		goal_block = state.gridworld.map[goal_loc[0]][goal_loc[1]]

	if goal_block == '.':
		goal_block = None

	# Queue of states to consider
	q = deque()
	q.append(state)

	# What is the previous location and inventory that got me here?
	pre_loc = dict()
	pre_loc[(state.agent_location, state.block_picked)] = None
	# What is the action that got me from the previous location and inventory to here?
	pre_action = dict()
	pre_action[(state.agent_location, state.block_picked)] = None

	found = False

	while q and not found:
		# Consider the first element in the queue
		curr_state = q.popleft()
		# if curr_state.agent_location == goal_loc and og_picked:
		#     print("At goal location")
		#     pdb.set_trace()
		# if md_check(curr_state.agent_location, goal_loc) and og_picked:
		#     print("1 away from goal loc")
		#     pdb.set_trace()
		# Go through neighbors
		action_space = list(construction.Action)
		for action in action_space:
			# Get the state & location of the neighbor
			next_state = transition(curr_state, action)
			next_loc = next_state.agent_location
			inv = next_state.block_picked

			# If we have not visited this state before
			if (next_loc, inv) not in pre_loc:
				# Add to the queue to consider
				q.append(next_state)
				# Save the action that gets us to that state and the state from which it gets us
				# there
				pre_loc[(next_loc, inv)] = (curr_state.agent_location, curr_state.block_picked)
				pre_action[(next_loc, inv)] = action
			found = (goal_loc, goal_block) in pre_loc

	if found:
		curr_loc, curr_inv = (goal_loc, goal_block)
		path = []
		while pre_action[(curr_loc, curr_inv)] is not None:
			path.insert(0, pre_action[(curr_loc, curr_inv)])
			curr_loc, curr_inv = pre_loc[(curr_loc, curr_inv)]
		return path
	else:
		return None


def grab_block(state, transition, goal_loc, useBFS=False):
	min_path_length = float("inf")
	action_to_path = {}
	for action in list(construction.Action):
	# @globalize
	# def findPath(action):
		state_ = transition(state, action)
		# path = bfs(state_, transition, goal_loc)
		# min_path_length = min(1 + len(path), min_path_length)
		if useBFS:
			# try:  # this only fails if we timeout
			# path = bfs(state_, transition, goal_loc)
			path = astar(state_, transition, goal_loc)
			if path is not None:
				path = len(path)
			else:
				path = float("inf")
			# except:
			# 	path = manhattan_distance(state_.agent_location, goal_loc)
		else:
			path = manhattan_distance(state_.agent_location, goal_loc)
		min_path_length = min(1 + path, min_path_length)
		action_to_path[action] = path

	# with multiprocessing.Pool() as pool:
	# 	pool.map(findPath, list(construction.Action))

	intercepting_actions = [a for a in list(construction.Action) if action_to_path[a] + 1 == min_path_length]
	values = [100-min_path_length] * len(intercepting_actions)

	# intercepting_actions = list(construction.Action)
	# actions = intercepting_actions
	# values = []
	# for a in actions:
	#     if action_to_path[a] + 1 == min_path_length:
	#         values.append(100-min_path_length)
	#     else:
	#         values.append(action_to_path[a])

	return intercepting_actions, values


def prevent_block_grab(state_0, transition, goal_loc, state_1):
	agent_0_loc = state_0.agent_location
	mid_x = (goal_loc[0] + agent_0_loc[0]) // 2
	mid_y = (goal_loc[1] + agent_0_loc[1]) // 2
	min_path_length = float("inf")
	action_to_path = {}
	for action in list(construction.Action):
		state_ = transition(state_1, action)
		# path = bfs(state_, transition, (mid_x, mid_y), goal_block='.')
		# min_path_length = min(1 + len(path), min_path_length)
		path = manhattan_distance(state_.agent_location, (mid_x, mid_y))
		min_path_length = min(1 + path, min_path_length)
		action_to_path[action] = path
	intercepting_actions = [a for a in list(construction.Action) if action_to_path[a] + 1 == min_path_length]
	values = [100-min_path_length] * len(intercepting_actions)

	# intercepting_actions = list(construction.Action)
	# actions = intercepting_actions
	# values = []
	# for a in actions:
	#     if action_to_path[a] + 1 == min_path_length:
	#         values.append(100-min_path_length)
	#     else:
	#         values.append(action_to_path[a])

	return intercepting_actions, values


def move_away_from(state, transition, L0_loc, colored_block_utilities):
	# to avoid getting trapped on border, find the shortest paths to quadrants the farthest away from L0
	num_rows, num_cols = state.gridworld.num_rows, state.gridworld.num_cols
	if L0_loc[0] < (num_rows // 2) and L0_loc[1] < (num_cols // 2):
		temp_rows = [int(num_rows * 0.25), int(num_rows * 0.75)]
		temp_cols = [int(num_cols * 0.75)]
	elif L0_loc[0] >= (num_rows // 2) and L0_loc[1] < (num_cols // 2):
		temp_rows = [int(num_rows * 0.25), int(num_rows * 0.75)]
		temp_cols = [int(num_cols * 0.25)]
	elif L0_loc[0] >= (num_rows // 2) and L0_loc[1] >= (num_cols // 2):
		temp_rows = [int(num_rows * 0.25)]
		temp_cols = [int(num_cols * 0.25), int(num_cols * 0.75)]
	else:
		temp_rows = [int(num_rows * 0.75)]
		temp_cols = [int(num_cols * 0.25), int(num_cols * 0.75)]

	quadrant_locs = []
	for row in temp_rows:
		for col in temp_cols:
			quadrant_locs.append((row, col))

	# find shortest path to each of the distant quadrants
	action_to_path = {}
	min_path_length = float("inf")
	for target in quadrant_locs:
		for action in list(construction.Action):
			state_ = transition(state, action)
			if manhattan_distance(state_.agent_location, L0_loc) > 2:
				# path = bfs(state_, transition, target, goal_block=state.block_picked)
				if action == construction.Action.PUT_DOWN:
					path = manhattan_distance(state_.agent_location, target) + 2
				else:
					path = manhattan_distance(state_.agent_location, target)
				if path is not None:
					# min_path_length = min(1 + len(path), min_path_length)
					min_path_length = min(1 + path, min_path_length)
				if action not in action_to_path:
					action_to_path[action] = path
				else:
					if path is not None:
						if action_to_path[action] is None:
							action_to_path[action] = path
						elif action_to_path[action] > path:
							action_to_path[action] = path
						# elif len(action_to_path[action]) > len(path):
						#     action_to_path[action] = path
	if min_path_length == float("inf"):
		return [construction.Action.STOP], [-1]
	actions = [a for a in list(construction.Action) if (a in action_to_path and action_to_path[a] + 1 == min_path_length)]
	values = [100-min_path_length] * len(actions)


	# values = []
	# for a in actions:
	#     if action_to_path[a] + 1 == min_path_length:
	#         values.append(100-min_path_length)
	#     else:
	#         values.append(action_to_path[a])


	return actions, values


def move_grabbed_next_to(state, transition, goal_loc, seek_conflict=False, colored_block_utils=None, useBFS=False):
	# find all neighbors of goal_loc
	actions = list(construction.Action)

	grid_map = state.gridworld.map
	free_cells = []
	for action in actions:
		next_pos = construction.next_pos(goal_loc, action)
		if construction.in_bound(next_pos, state.gridworld):
			if grid_map[next_pos[0]][next_pos[1]] == '.' and manhattan_distance(next_pos, goal_loc) == 1:
				free_cells.append(next_pos)

	# find shortest path to each of the neighbors
	action_to_path = {}
	min_path_length = float("inf")
	for target in free_cells:
		for action in actions:
		# @globalize
		# def findPath(action):
			state_ = transition(state, action)

			if not useBFS:
				path = manhattan_distance(state_.agent_location, target)
				if action == construction.Action.PUT_DOWN or state_.agent_location == state.agent_location:
					path += 2
			else:
				# try:  # this only fails if we timeout
				# path = bfs(state_, transition, target, goal_block=state.block_picked)
				path = astar(state_, transition, goal_loc, goal_block=state.block_picked)
				if path is not None:
					path = len(path)
				else:
					path = float("inf")
				# except:
				# 	path = manhattan_distance(state_.agent_location, target)
				# 	if action == construction.Action.PUT_DOWN or state_.agent_location == state.agent_location:
				# 		path += 2

			if path is not None:
				min_path_length = min(1 + path, min_path_length)

			if action not in action_to_path:
				action_to_path[action] = path
			else:
				if path is not None:
					if action_to_path[action] is None:
						action_to_path[action] = path
					elif action_to_path[action] > path:
						action_to_path[action] = path

		# with multiprocessing.Pool() as pool:
		# 	pool.map(findPath, list(actions))


	# actions = [a for a in list(construction.Action) if len(action_to_path[a]) + 1 == min_path_length]
	actions = [a for a in actions if action_to_path[a] + 1 == min_path_length]

	block_1, block_2 = max(colored_block_utils, key=colored_block_utils.get)
	colored_blocks = state.colored_blocks
	if md_check(colored_blocks[block_1], colored_blocks[block_2]) and not seek_conflict:
		values = [100] * len(actions)
	elif md_check(colored_blocks[block_1], colored_blocks[block_2]) and seek_conflict:
		values = [-100] * len(actions)
	# elif not md_check(colored_blocks[block_1], colored_blocks[block_2]) and seek_conflict:
	#     values = [100-min_path_length] * len(actions)
	# else:
	#     values = [100-min_path_length] * len(actions)
	else:
		values = []
		for a in actions:
			if action_to_path[a] + 1 == min_path_length:
				values.append(100-min_path_length)

	return actions, values

# TODO: See how picking up blue block affects prediction for green-yellow pair under parameters False, 50, 250
def determine_subgoals(state_0, transition, colored_block_utilities, state_1=None, agent_id=0, seek_conflict=True, useBFS=False):
	if agent_id == 0:  # L0/L2 case
		colored_blocks = state_0.colored_blocks
		inv = state_0.block_picked
		block_1, block_2 = max(colored_block_utilities, key=colored_block_utilities.get)
		best_util = colored_block_utilities[(block_1, block_2)]
		if inv == block_1:
			if md_check(colored_blocks[block_1], colored_blocks[block_2]):
				actions_ = [Action.PUT_DOWN]
				values_ = [best_util - 1]
			else:
				actions_, values_ = move_grabbed_next_to(state_0, transition, colored_blocks[block_2], colored_block_utils=colored_block_utilities, useBFS=useBFS)
		elif inv == block_2:
			if md_check(colored_blocks[block_1], colored_blocks[block_2]):
				actions_ = [Action.PUT_DOWN]
				values_ = [best_util - 1]
			else:
				actions_, values_ = move_grabbed_next_to(state_0, transition, colored_blocks[block_1], colored_block_utils=colored_block_utilities, useBFS=useBFS)
		else:
			if inv is not None:  # drop a block sub goal
				actions_ = [construction.Action.PUT_DOWN]
				values_ = [10]
			else:
				actions_1, values_1 = grab_block(state_0, transition, colored_blocks[block_1], useBFS=useBFS)
				actions_2, values_2 = grab_block(state_0, transition, colored_blocks[block_2], useBFS=useBFS)

				actions_ = actions_1 + actions_2
				values_ = values_1 + values_2
		actions_.append(construction.Action.STOP)
		values_.append(-1)  # stopping in place
		return actions_, values_
	else:  # L1 case
		assert state_1 is not None
		colored_blocks = state_1.colored_blocks
		inv = state_1.block_picked
		block_1, block_2 = max(colored_block_utilities, key=colored_block_utilities.get)
		best_util = colored_block_utilities[(block_1, block_2)]
		if inv is not None:  # have an item picked
			if not seek_conflict:  # if not seeking conflict, bring block in ideal pair to L0, or put down unnecessary blocks
				if inv == block_1 or inv == block_2:
					if md_check(state_1.agent_location, state_0.agent_location):
						actions_ = [construction.Action.PUT_DOWN]
						values_ = [best_util-1]
					else:
						actions_, values_ = move_grabbed_next_to(state_1, transition, state_0.agent_location, seek_conflict=seek_conflict, colored_block_utils=colored_block_utilities)
				else:
					actions_, values_ = [construction.Action.PUT_DOWN], [best_util - 1]
			else:  # if seeking conflict, runaway or prevent block grabs
				if inv == block_1 or inv == block_2:
					actions_, values_ = move_away_from(state_1, transition, state_0.agent_location, colored_block_utilities)
				else:
					if state_0.block_picked == block_1:
						actions_1, values_1 = prevent_block_grab(state_0, transition, colored_blocks[block_2], state_1)
						actions_2, values_2 = [construction.Action.PUT_DOWN], [best_util-1]
						# actions_2, values_2 = grab_block(state_1, transition, colored_blocks[block_2])
					elif state_0.block_picked == block_2:
						actions_1, values_1 = prevent_block_grab(state_0, transition, colored_blocks[block_1], state_1)
						actions_2, values_2 = [construction.Action.PUT_DOWN], [best_util-1]
						# actions_2, values_2 = grab_block(state_1, transition, colored_blocks[block_1])
					else:
						actions_1, values_1 = prevent_block_grab(state_0, transition, colored_blocks[block_2], state_1)
						actions_2, values_2 = prevent_block_grab(state_0, transition, colored_blocks[block_1], state_1)
					actions_ = actions_1 + actions_2
					values_ = values_1 + values_2
					actions_, values_ = [Action.PUT_DOWN], [best_util-1]  # just focus on dropping irrelevant blocks
		else:  # no item picked
			if not seek_conflict:  # if trying to help, either move a block to L0
				if state_1.agent_location == colored_blocks[block_1] or state_1.agent_location == colored_blocks[block_2] and \
						md_check(colored_blocks[block_1], colored_blocks[block_2]):
					actions_, values_ = move_away_from(state_1, transition, state_1.agent_location, colored_block_utilities)
				else:
					md_1_1 = manhattan_distance(state_1.agent_location, colored_blocks[block_1])
					md_1_2 = manhattan_distance(state_1.agent_location, colored_blocks[block_2])
					md_0_1 = manhattan_distance(state_0.agent_location, colored_blocks[block_1])
					md_0_2 = manhattan_distance(state_0.agent_location, colored_blocks[block_2])
					if md_0_1 > md_1_1 and md_0_2 <= md_1_2:  # L1 is closer to block 1 and L2 is closer to block 2
						actions_, values_ = grab_block(state_1, transition, colored_blocks[block_1])
					elif md_0_1 <= md_1_1 and md_0_2 > md_1_2:  # L1 is closer to block 2 and L2 is closer to block 1
						actions_, values_ = grab_block(state_1, transition, colored_blocks[block_2])
					else:
						if md_1_1 < md_1_2:  # L1 is closer to block 1 than L1 is to block 2
							actions_, values_ = grab_block(state_1, transition, colored_blocks[block_2])
						else:
							actions_, values_ = grab_block(state_1, transition, colored_blocks[block_1])
			else:  # if trying to hurt
				if state_0.block_picked == block_1:
					actions_1, values_1 = prevent_block_grab(state_0, transition, colored_blocks[block_2], state_1)
					actions_2, values_2 = grab_block(state_1, transition, colored_blocks[block_2])
				elif state_0.block_picked == block_2:
					actions_1, values_1 = prevent_block_grab(state_0, transition, colored_blocks[block_1], state_1)
					actions_2, values_2 = grab_block(state_1, transition, colored_blocks[block_1])
				else:
					actions_1, values_1 = prevent_block_grab(state_0, transition, colored_blocks[block_1], state_1)
					actions_2, values_2 = prevent_block_grab(state_0, transition, colored_blocks[block_2], state_1)
				actions_ = actions_1 + actions_2
				values_ = values_1 + values_2
			# if values_.count(max(values_)): # if the L1 agent isn't sure what to do it should wait
			#     actions_ = [construction.Action.STOP]
			#     values_ = [0]
		temp = {}
		for i, action in enumerate(actions_):
			if action not in temp:
				temp[action] = values_[i]
			else:
				temp[action] += values_[i]

		temp[construction.Action.STOP] = -1     # add stopping as a valid action        

		return list(temp.keys()), list(temp.values())




def plan_shortest_path(state, transition, colored_block_utilities):
	"""Shortest-path based planner.

	Returns
		actions [List[Action]]: list of possible first actions that lead to a path with the
			shortest length
		value (int): computed as goal_utility - shortest_path_length
	"""
	# Find goal location
	colored_blocks = state.colored_blocks
	colored_block_utilities_on_the_map = {}
	for k, v in colored_block_utilities.items():
		b1, b2 = k
		if b1 in colored_blocks and b2 in colored_blocks:
			colored_block_utilities_on_the_map[k] = v

	goal_pair = max(colored_block_utilities_on_the_map, key=colored_block_utilities_on_the_map.get)  # best goal pair
	goal_1, goal_2 = goal_pair
	best_util = colored_block_utilities[goal_pair]  # utility of the pair
	loc1, loc2 = colored_blocks[goal_1], colored_blocks[goal_2]  # location of each block in the pair

	# check if we already reached the goal
	distance_check = md_check(loc1, loc2)
	inv_check = state.block_picked is None

	if distance_check and not inv_check:  # right location but need to clear inventory to reach terminal state
		return [Action.PUT_DOWN], [best_util]
	elif distance_check and inv_check:
		return [Action.STOP], [best_util]
	else:
		inv = state.block_picked
		cell_1 = state.gridworld.map[loc1[0]][loc1[1]]
		cell_2 = state.gridworld.map[loc2[0]][loc2[1]]
		# case where the first block is selected by either agent, go to second block
		if inv == goal_1 or (cell_1 == '.' and inv is None):
			min_path_length = float('inf')
			path_from_action = {}
			next_actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN, Action.STOP]
			random.shuffle(next_actions)
			for action in next_actions:  # don't want to put down block
				state_ = pickle.loads(pickle.dumps(state))
				new_state = transition(state_, action)
				# pdb.set_trace()
				path = bfs(new_state, transition, loc2)
				path_from_action[action] = path
				if path is not None:
					min_path_length = min(len(path) + 1, min_path_length)
				# pdb.set_trace()
			if min_path_length == float('inf'):  # haven't found a path, wait
				return [Action.STOP], [best_util - min_path_length]
			actions = []
			for action, path in path_from_action.items():
				if path is not None and min_path_length == len(path) + 1:
					actions.append(action)
			return actions, [best_util - min_path_length] * len(actions)
		# case where the second block is selected by either agent, go to first block
		elif inv == goal_2 or (cell_2 == '.' and inv is None):
			min_path_length = float('inf')
			path_from_action = {}
			next_actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN, Action.STOP]  # don't want to put down block
			for action in next_actions:
				state_ = pickle.loads(pickle.dumps(state))
				new_state = transition(state_, action)
				# pdb.set_trace()
				path = bfs(new_state, transition, loc1)  # shortest path to missing block
				path_from_action[action] = path
				if path is not None:
					min_path_length = min(len(path) + 1, min_path_length)
				# pdb.set_trace()
			if min_path_length == float('inf'):  # haven't found a path, wait
				return [Action.STOP], [best_util - min_path_length]
			actions = []
			for action, path in path_from_action.items():
				if path is not None and min_path_length == len(path) + 1:
					actions.append(action)
			return actions, [best_util - min_path_length] * len(actions)
		else:  # either no blocks are selected so chase either, or both are selected so bring together
			goal_md = manhattan_distance(loc1, loc2)
			# print(goal_md)
			path1_from_action = {}
			action_space = list(Action)
			min_path_length1 = float('inf')
			for action in action_space:
				state_ = pickle.loads(pickle.dumps(state))
				new_state = transition(state_, action)
				path = bfs(new_state, transition, loc1)  # shortest path to missing block 1
				path1_from_action[action] = path
				if path is not None:
					min_path_length1 = min(len(path) + 1, min_path_length1)
			next_actions = []
			action_value = []
			for action, path in path1_from_action.items():
				if path is not None and min_path_length1 == len(path) + 1:
					# print([action] + path)
					next_actions.append(action)
					action_value.append(best_util - min_path_length1 - goal_md)
			path2_from_action = {}
			min_path_length2 = float("inf")
			for action in action_space:
				state_ = pickle.loads(pickle.dumps(state))
				new_state = transition(state_, action)
				path = bfs(new_state, transition, loc2)  # shortest path to missing block 2
				path2_from_action[action] = path
				if path is not None:
					min_path_length2 = min(len(path) + 1, min_path_length2)
			for action, path in path2_from_action.items():
				if path is not None and min_path_length2 == len(path) + 1:
					# print([action] + path)
					next_actions.append(action)
					action_value.append(best_util - min_path_length2 - goal_md)
			if min_path_length1 == float("inf") and min_path_length2 == float("inf"):
				return [Action.STOP], [best_util - min(min_path_length1, min_path_length2) - goal_md]
			else:
				return next_actions, action_value


class AgentL0:
	"""Level-0 agent

	Args
		gridworld (envs.construction.Gridworld)
		colored_block_utilities (dictionary)
			key (tuple (str, str)) is a colored block pair
			value (int) is the utility of that block pair

			Contains utilities for all possible colored block pairs in the map
			len(colored_block_utilities) can be > number of colored blocks pair combinations in the gridworld
		transition (function)
			input: state (envs.construction.State), action (envs.construction.Action)
			output: next_state (envs.construction.State)
		beta (constant coefficient for noisy Boltzmann policy)
	"""

	def __init__(self, gridworld, colored_block_utilities, transition, beta=0.01):
		self.observations = []
		self.actions = []
		self.gridworld = gridworld
		self.colored_block_utilities = colored_block_utilities
		self.transition = transition
		self.beta = beta
		# assert self.beta == 10.0

		self.num_possible_block_pairs = len(self.colored_block_utilities)
		self.num_colored_block_locations = len(self.gridworld.get_colored_block_locations())

		self.action_history = set([])


	def clone(self):
		return AgentL0(self.gridworld, self.colored_block_utilities, self.transition, beta=self.beta)

	@property
	def num_observations(self):
		return len(self.observations)

	def get_belief(self):
		# actual state with probability of 1
		obs = self.observations[-1]
		return [(obs, 1)]

	def get_action_probs(self, belief=None, useBFS=False):
		if belief is None:
			belief = self.get_belief()
		action_space = list(construction.Action)
		uniform_probs = np.ones((len(action_space),)) / len(action_space)
		if belief is None:
			return uniform_probs
		else:
			expected_utility = {a: None for a in action_space}
			for (state, prob) in belief:
				actions, values = determine_subgoals(state, self.transition, self.colored_block_utilities, useBFS=useBFS)
				# print(state)
				# print(actions)
				# print(values)
				for i, action in enumerate(actions):
					if expected_utility[construction.Action(action.value)] is None:
						expected_utility[construction.Action(action.value)] = 0
					expected_utility[construction.Action(action.value)] += prob * values[i]
			# print(f"L0 expected utility for utility = {self.colored_block_utilities}: {expected_utility}")
			# print("Expected utility: ")
			# print(expected_utility)
			# print(self.colored_block_utilities)
			# print("-----------")
			action_log_probs = np.full((len(action_space),), -1e6)
			for action_id, action in enumerate(action_space):
				if expected_utility[action] is not None:
					action_log_probs[action_id] = self.beta * expected_utility[action]
			if scipy.special.logsumexp(action_log_probs) < np.log(1e-6):
				return uniform_probs
			else:
				action_log_probs_normalized = action_log_probs - scipy.special.logsumexp(
					action_log_probs
				)
				action_probs_normalized = np.exp(action_log_probs_normalized)
				action_probs_normalized = action_probs_normalized / np.sum(action_probs_normalized)
				# print(f"Utility for L0: {self.colored_block_utilities}")
				# for i, action in enumerate(action_space):
				#     print(f"P({action} for L0 | g): {action_probs_normalized[i]}")
				# print("-------")
				if np.isnan(action_probs_normalized).any():
					raise RuntimeError("nan action probs")
				# pdb.set_trace()
				return action_probs_normalized

	def get_action(self, observation=None, return_info=False, useBFS=True):
		if observation is not None:
			self.observations.append(observation)

		belief = self.get_belief()
		action_probs = self.get_action_probs(belief=belief, useBFS=useBFS)
		action_space = list(construction.Action)
		max_actions = [a for i, a in enumerate(action_space) if action_probs[i] == max(action_probs)]
		random.shuffle(max_actions)
		# action = np.random.choice(action_space, p=action_probs)  #  uncomment for stochastic L0
		# mu = random.random()
		# mu = 1.0
		# if mu <= (1/20):
		#     action = np.random.choice(action_space)
		# else:
		action = np.random.choice(max_actions)  # pick a random maximizing action

		if return_info:
			return action, {"belief": belief, "action_probs": action_probs}
		else:
			return action
