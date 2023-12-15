import scipy.special
import _pickle as pickle
from enum import Enum
from collections import deque
import itertools
import numpy as np
import random
import torch
from randomStreetGen import isVisible
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point, Line
import pdb
# from pathos.multiprocessing import ProcessingPool as Pool

class Action(Enum):
	FORWARD = (0, 4000)
	LEFT = (0.5, 0)
	RIGHT = (-1.15, -1100)
	STOP = (0, -1e7)
	SIGNAL = (0, 0)
# class Action(Enum):
# 	FORWARD = (0, 4000)
# 	LEFT = (0.75, -300)
# 	RIGHT = (-0.69, -150)
# 	STOP = (0, -1e7)
# 	SIGNAL = (0, 0)

# class Action(Enum):
# 	FORWARD = (0, 5000)
# 	LEFT = (0.42, -50)
# 	RIGHT = (-0.9, -170)
# 	STOP = (0, -1e7)
# 	SIGNAL = (0, 0)


pair1 = [0, 1]
pair2 = [2, 3]
pair3 = [4, 5]
pair4 = [6, 7]
pair5 = [8, 9]
pair6 = [10, 11]
pair7 = [12, 13]
pair8 = [14, 15]
all_pairs = [pair1, pair2, pair3, pair4, pair5, pair6, pair7, pair8]

def laneImportance(id, goal):
	# importance is defined as any lane in which a car there could intersect with the agent's path
	if goal == "forward":
		if id in pair1:
			return pair3 + pair8 + pair6 + pair5
		elif id in pair3:
			return pair1 + pair6 + pair8 + pair7
		elif id in pair6:
			return pair1 + pair3 + pair8 + pair2
		elif id in pair8:
			return pair1 + pair6 + pair3 + pair4
	elif goal == "left":
		if id in pair1:
			return pair3 + pair8 + pair6 + pair4
		elif id in pair3:
			return pair1 + pair6 + pair8 + pair5
		elif id in pair6:
			return pair1 + pair3 + pair8 + pair7
		elif id in pair8:
			return pair1 + pair6 + pair3 + pair2
	elif goal == "right":
		if id in pair1:
			return pair3 + pair7 + pair6 
		elif id in pair3:
			return pair2 + pair6 + pair8
		elif id in pair6:
			return pair1 + pair4 + pair8
		elif id in pair8:
			return pair1 + pair5 + pair3
	return []  # for stopping or all other cases, no lane matters

def goalToLoc(id, goalString, idToLoc):
	if goalString == "forward":
		if id in pair6 or id in pair2:
			return idToLoc[3], 3
		elif id in pair3 or id in pair7:
			return idToLoc[13], 13
		elif id in pair8 or id in pair4:
			return idToLoc[7], 7
		else:
			return idToLoc[9], 9
	elif goalString == "right":
		if id in pair6:
			return idToLoc[7], 7
		elif id in pair3:
			return idToLoc[3], 3
		elif id in pair1:
			return idToLoc[13], 13
		elif id in pair8:
			return idToLoc[9], 9
	elif goalString == "left":
		if id in pair1:
			return idToLoc[7], 7
		elif id in pair3:
			return idToLoc[9], 9
		elif id in pair6:
			return idToLoc[13], 13
		elif id in pair8:
			return idToLoc[3], 3
	elif goalString == "stop":
		return idToLoc[id], id
	return goalToLoc(id, "forward", idToLoc)  # already completed turn so now we move forward

def locToGoalString(id, laneSpotID):
	if id == laneSpotID:
		return "stop"
	elif (id in pair6 or id in pair2) and laneSpotID == 3:
		return "forward"
	elif (id in pair3 or id in pair7) and laneSpotID == 13:
		return "forward"
	elif (id in pair8 or id in pair4) and laneSpotID == 7:
		return "forward"
	elif id in pair6 and laneSpotID == 7:
		return "right"
	elif id in pair3 and laneSpotID == 3:
		return "right"
	elif id in pair1 and laneSpotID == 13:
		return "right"
	elif id in pair8 and laneSpotID == 9:
		return "right"
	elif id in pair1 and laneSpotID == 7:
		return "left"
	elif id in pair3 and laneSpotID == 9:
		return "left"
	elif id in pair6 and laneSpotID == 13:
		return "left"
	elif id in pair8 and laneSpotID == 3:
		return "left"
	else:
		return "forward"



def carInFront(agentID, world):
	if agentID in [0, 3, 4, 7, 13, 14, 9, 10]:  # even ID cars can't have a car in front of them
		return False, None
	else:
		for car in world.dynamic_agents:
			if agentID % 2 == 0:
				if car.ID == agentID + 1:  # if we have a car in front of us, return the car's velocity
					return True, pickle.loads(pickle.dumps(car.velocity))
			else:
				if car.ID == agentID - 1:  # if we have a car in front of us, return the car's velocity
					return True, pickle.loads(pickle.dumps(car.velocity))
		return False, None  # no car exists in front


class AgentL0:
	'''
	L0 agent gets samples beliefs about POMDP and takes best actions assuming other cars
	don't change their behaviors

	Goal action is overall goal to move forward, turn left, turn right, or stop
		NOT THE SAME AS CONTROL ACTIONS
	'''
	def __init__(self, agentID, goalActionString, transition, initialLocMapping, carExistPrior=0.5, beta=0.001,
		sampled_actions=2, lookAheadDepth=2, state_model=None, exist_model=None, inference_algorithm=None):
		self.agentID = agentID

		self.transition = transition

		self.inference_algorithm = inference_algorithm

		# partial observations from the world, each one should be a CARLO world object
		# assume all cars have a distinct integer representation
		self.observations = []  
		self.full_action_history = []  # full action history for all agents in the world, unobserved actions should be pruned


		# maps carID (int) to fixed potential initial locations on map Point(x, y)
		self.initialLocMapping = initialLocMapping
		self.num_possible_cars = len(initialLocMapping)

		self.goalActionString = goalActionString
		self.goalAction, self.targetSlot = goalToLoc(self.agentID, self.goalActionString, self.initialLocMapping)  # high level action goal (really it's a Point(x, y) target location)
		self.beta = beta

		self.sampled_actions_per = sampled_actions
		self.lookAheadDepth = lookAheadDepth

		# probability for an unobserved car to exist
		self.carExistPrior = carExistPrior

		self.state_model = state_model
		if self.state_model is not None:
			self.state_model.eval()

		self.exist_model = exist_model
		if self.exist_model is not None:
			self.exist_model.eval()


	@property
	def num_observations(self):
		return len(self.observations)

	'''
	We need to pass just the observation (World) through the network
	'''
	def get_nn_state_belief(self):
		# encoded actions for invisible agents will be disregarded in tensor representation
		state_action_tensors = torch.stack([
			state_action_to_joint_tensor(self.observations[i], self.full_action_history[i], device=self.state_model.device)
			for i in range(len(self.full_action_history))
		]).unsqueeze(0)
		lens = torch.LongTensor([s.shape[0] for s in state_action_tensors]).cpu()
		# get prediction for current timestep
		with torch.no_grad():
			log_prob_exist = self.exist_model(state_action_tensors, lens)[-16:]  # the last 16 cars should be our current view
			prob_exist = torch.softmax(log_prob_exist, 1).cpu().detach().numpy()
			prob_exist /= prob_exist.sum(axis=1, keepdims=True)

			telemetry_pred = self.state_model(state_action_tensors, lens)[-16:].cpu().detach().numpy()


		# this is a fast operation to determine which cars to sample
		importantCars = laneImportance(self.agentID, self.goalActionString)

		# sample relevant cars
		belief = self.observations[-1].clone()
		for car in belief.dynamic_agents:
			if car.ID != self.agentID:
				belief.dynamic_agents.remove(car)
			# elif car.speed == 0:
			# 	car.speed = telemetry_pred[car.ID][3]  # sample speed for self if you are thought to be still
		prob = 1.0
		for importantID in importantCars:  # the network should know which cars to pay attention to and which are worth sampling
			# sample whether that car exists
			existence_prob = prob_exist[importantID]
			if self.inference_algorithm == 'random':
				existence_prob = [0.5, 0.5]
			sampledCar = np.random.choice([False, True], p=existence_prob)
			prob *= existence_prob[int(sampledCar)]  # what is the probability of sampling/not sampling this car

			if sampledCar:
				telemetry = telemetry_pred[importantID]

				carLoc = Point(telemetry[0], telemetry[1])
				carAngle = idToHeadingAngle(importantID)
				if importantID in [4,5,6,7,12,13,15,14]:
					color = "blue"
				else:
					color = "red"
				car_sample = Car(carLoc, carAngle, ID=importantID, color=color)
				car_sample.velocity = Point(telemetry[2],0)
				belief.add(car_sample)

		return [(belief, prob)], None


	def get_belief(self):
		# can we amortize belief
		if self.state_model is not None and self.exist_model is not None and len(self.full_action_history) > 0:
			beliefs = []
			for _ in range(1):
				beliefs.append(self.get_nn_state_belief()[0][0])
			return beliefs, None


		partialObs = self.observations[-1]

		idVelocityPairs = []  # stores tuple of (id, velocity, probability of existing)

		currLoc = None  # where is our car currently
		visibleIDs = []  # which cars can we see
		for car in partialObs.dynamic_agents:  # assume no pedestrians
			visibleIDs.append(car.ID)  # so we don't overwrite it
			idVelocityPairs.append((car.ID, car.center.x, car.center.y, car.speed, 1.0))  # 100 percent chance of seeing what we see
			if car.ID == self.agentID:
				currLoc = car

		nonVisibleIDs = []
		for id, loc in self.initialLocMapping.items():
			if id in visibleIDs:  # don't want to overwrite cars we've seen
				continue
			if isVisible(currLoc, loc, partialObs):  # we can see empty space, meaning no car
				visibleIDs.append(id)
				idVelocityPairs.append((id, loc.x, loc.y, 0.0, 0.0))
			else:
				nonVisibleIDs.append(id)

		# remove any un-important ids so we don't waste samples
		importantIds = laneImportance(self.agentID, self.goalActionString)
		idToSample = []
		for id in nonVisibleIDs:
			if id in importantIds:
				idToSample.append(id)
			else:
				x,y = self.initialLocMapping[id].x, self.initialLocMapping[id].y
				idVelocityPairs.append((id, x, y, 0.0, 0.0))  # give unimportant non visible stuff a 50% chance of existing

		for pair in all_pairs:
			# if we have the lead car invisible, no point sampling one behind it
			if idToSample.count(pair[0]) + idToSample.count(pair[1]) == 2:
				idToSample.remove(pair[1])
				x,y = self.initialLocMapping[pair[1]].x, self.initialLocMapping[pair[1]].y
				idVelocityPairs.append((pair[1], x, y, 0.0, 0.0))  # give a random chance of appearing to unimportant

		# random.shuffle(idToSample)  # do this to avoid repeating order

		'''
		now that we've marked the ids of all cars we believe to exist, 
		we need to create belief state
		'''


		def buildBelief(id, idsToInclude, prob):
			if id == len(idToSample):
				# start by cloning partial observation so we don't affect it
				belief = pickle.loads(pickle.dumps(partialObs)) 
				# all of the visible elements are already included in the environment
				for sampledID in idsToInclude:  # add cars we sample to exist
					carLoc = pickle.loads(pickle.dumps(self.initialLocMapping[sampledID]))
					carAngle = idToHeadingAngle(sampledID)
					if sampledID in [4,5,6,7,12,13,15,14]:
						color = "blue"
					else:
						color = "red"
					sampledCar = Car(carLoc, carAngle, ID=sampledID, color=color)
					behindCar, initVelocity = carInFront(sampledID, partialObs)
					# add sampled car to the world with forward movement
					if behindCar:  # if we are behind a car, copy it's velocity
						sampledCar.velocity = initVelocity
					else:  # otherwise sample some random velocity
						min_speed = 50
						max_speed = 50
						sampledCar.velocity = Point(random.uniform(min_speed, max_speed),0)

					if len(idsToInclude) == len(idToSample):  # want to store so we can recreate
						idVelocityPairs.append((sampledCar.ID, sampledCar.center.x, sampledCar.center.y, sampledCar.speed, self.carExistPrior))

					belief.add(sampledCar)
				return [(belief, prob)]  # probability of generating this state
			else:
				idsCopy = pickle.loads(pickle.dumps(idsToInclude))
				idsCopy.append(idToSample[id])
				included = buildBelief(id + 1, idsCopy, prob * self.carExistPrior)  # tree if car exists
				excluded = buildBelief(id + 1, idsToInclude, prob * (1 - self.carExistPrior))  # tree if car doesn't exist
				return included + excluded  # combine the lists

		belief_distrib = buildBelief(0, [], 1.0)  # full belief tree of variable length


		belief_tensor = [[]] * self.num_possible_cars
		for cID, x, y, speed, existProb in idVelocityPairs:
			belief_tensor[cID] = [existProb, x, y, speed]

		return belief_distrib, belief_tensor

	def get_action_probs(self, belief=None):
		if belief is None:
			belief, belief_tensor = self.get_belief()
		action_space = list(Action)
		uniform_probs = np.ones((len(action_space),)) / len(action_space)
		if belief is None:
			return uniform_probs
		else:
			# sample num_samples trajectories of length lookAheadDepth
			# for each trajectory
				# trajectory value = 0
				# for each belief
					# get value of trajectorybased on belief
					# aggregate trajectory based on probability of belief
			# pick first action from trajectory with highest value




			expected_utility = {a: None for a in action_space}

			actions, values = determine_subgoals(belief, self.transition, self.agentID, self.goalAction, self.goalActionString, self.lookAheadDepth)

			for i, action in enumerate(actions):
				expected_utility[action] = values[i]


			# for (state, prob) in belief:
			# 	# actions, values = determine_subgoals(state, self.transition, self.agentID, self.goalAction)
			# 	actions, values = cross_entropy_planning(state, self.transition, self.agentID, self.goalAction, 
			# 		self.lookAheadDepth, self.sampled_actions_per, self.goalActionString)
			# 	for i, action in enumerate(actions):
			# 		if action.value == Action.RIGHT.value and self.goalActionString != "right":
			# 			pdb.set_trace()
			# 		if expected_utility[Action(action.value)] is None:
			# 			expected_utility[Action(action.value)] = 0
			# 		expected_utility[Action(action.value)] += prob * values[i]

			action_log_probs = np.full((len(action_space),), -1e6)
			for action_id, action in enumerate(action_space):
				# if self.goalActionString != "right":
				# 	try:
				# 		assert expected_utility[Action.RIGHT] is None
				# 	except:
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
				if np.isnan(action_probs_normalized).any():
					raise RuntimeError("nan action probs")
				return action_probs_normalized

	def get_action(self, observation=None, return_info=False, prev_actions = None):
		if observation is not None:
			self.observations.append(observation)
		if prev_actions is not None:
			self.full_action_history.append(prev_actions)

		belief, belief_tensor = self.get_belief()  # get prob distribution over possible states
		action_probs = self.get_action_probs(belief=belief)
		action_space = list(Action)
		max_actions = [a for i, a in enumerate(action_space) if action_probs[i] == max(action_probs)]
		random.shuffle(max_actions)
		action = np.random.choice(max_actions)  # pick a random maximizing action
		if self.goalActionString != "right" and action == Action.RIGHT:
			pdb.set_trace()
			action_probs = self.get_action_probs(belief=belief)
			
		if return_info:
			return action, {"belief": belief, "belief_tensor": belief_tensor, "action_probs": action_probs}
		else:
			return action


def determine_subgoals(beliefs, transition, agentID, goalActionLoc, goalActionString, lookAheadDepth=5):
	'''
	beliefs is list of (Carlo World, prob) from get_beliefs
	transition is the transition function dynamics of the world
		input: state of World type, actionDict mapping {agentID: action}
		output: next_state
	agentID: is the current agent we're controlling 
	goalActionString: is one of [forward, left, right]
	'''
	action_space = list(Action)


	def best_solo_plan(state):
		if goalActionString == 'forward':
			suggested_action = Action.FORWARD
		else:
			nearWall = state.agent_near_wall(agentID)
			if nearWall or state.agent_speed(agentID) == 0.0:  # wall to the right or stopped so move forward
				suggested_action = Action.FORWARD
			elif goalActionString == 'right':  # not near wall so turn right
				suggested_action = Action.RIGHT
			else:  # not near wall so turn left
				suggested_action = Action.LEFT
		return suggested_action




	base_action_value = 100
	suggested_action = None


	crashProb = 0.0

	for belief, prob in beliefs:
		curr_state = pickle.loads(pickle.dumps(belief))
		detectedCrash = False
		for t in range(lookAheadDepth):
			# assume other agents are going to maintain their previous actions
			action_dict = {}
			for other_agent in belief.dynamic_agents:
				if other_agent.ID != agentID:
					action_dict[other_agent.ID] = other_agent.prev_action
			action_dict[agentID] = best_solo_plan(belief)

			if suggested_action is None:  # for first timestep only
				suggested_action = action_dict[agentID]

			next_state = transition(curr_state, action_dict)
			nextAgent = None 
			for temp_a in next_state.dynamic_agents:  # looking for which agent we are
				if temp_a.ID == agentID:
					nextAgent = temp_a
					break

			if next_state.collision_exists(nextAgent):
				crashProb += ((0.99 ** t) * prob)
				break

			curr_state = next_state



	# if random.random() <= crashProb:
	# 	return [Action.STOP], [100]
	# else:
	# 	return [suggested_action], [100]
	if crashProb >= 0.5:
		return [Action.STOP], [100]
	else:
		return [suggested_action], [100]

	# stopValue = crashProb * base_action_value
	# suggested_action_value = (1-crashProb) * base_action_value
	# actions = [suggested_action, Action.STOP]
	# values = [suggested_action_value, stopValue]

	# return actions, values

def cross_entropy_planning(belief_state, transition, agentID, goalActionLoc, lookAheadDepth=5, num_actions_sample=2, goalActionString="forward", multiprocess=False):
	action_space = list(Action)
	if goalActionString == "forward":
		action_space.remove(Action.RIGHT)
		action_space.remove(Action.LEFT)
	elif goalActionString == "right":
		action_space.remove(Action.LEFT)
	elif goalActionString == "left":
		action_space.remove(Action.RIGHT)



	# assume other agents are going to maintain their previous actions
	action_dict = {}
	for other_agent in belief_state.dynamic_agents:
		if other_agent.ID != agentID:
			action_dict[other_agent.ID] = other_agent.prev_action

	def sample_lookahead(curr_state):
		firstAction = None
		value = 0.0
		currAgent = None
		for t in range(lookAheadDepth):
			temp_action_dict = pickle.loads(pickle.dumps(action_dict))
			action = random.choice(action_space)
			temp_action_dict[agentID] = action
			next_state = transition(curr_state, temp_action_dict)
			nextAgent = None 
			for temp_a in next_state.dynamic_agents:  # looking for which agent we are
				if temp_a.ID == agentID:
					nextAgent = temp_a
					break

			if firstAction is None:
				firstAction = action

			if action.value != Action.STOP.value and currAgent is not None:
				currLoc = (currAgent.center.x, currAgent.center.y)
				nextLoc = (nextAgent.center.x, nextAgent.center.y)

				# nextCars = [(a.ID, a.center.x, a.center.y) for a in next_state.dynamic_agents]
				# oldCars = [(a.ID, a.center.x, a.center.y) for a in curr_state.dynamic_agents]
				# pdb.set_trace()
				try:
					assert currLoc != nextLoc
				except:
					pdb.set_trace()

			value -= goalActionLoc.distanceTo(nextAgent.obj)  # negative distance is state value

			if next_state.collision_exists(nextAgent):
				value -= 100
				break

			currAgent = nextAgent
			curr_state = next_state

		return (firstAction, value)


	actionExpectation = {}
	state_copies = [belief_state.clone()] * num_actions_sample
	if not multiprocess:
		for s in state_copies:
			action, value = sample_lookahead(s)
			if action not in actionExpectation:
				actionExpectation[action] = value
			else:
				actionExpectation[action] += value
	else:
		with Pool() as pool:
			for (action, value) in pool.map(sample_lookahead, state_copies):
				if action not in actionExpectation:
					actionExpectation[action] = value
				else:
					actionExpectation[action] += value

	max_action = max(actionExpectation, key=actionExpectation.get)
	maxActionValue = actionExpectation[max_action]

	actions = [a for a in actionExpectation if actionExpectation[a] == maxActionValue]
	values = [maxActionValue] * len(actions)
	return actions, values



def idToHeadingAngle(aID):
	if aID in [0, 1, 8, 9]:
		return -np.pi/2
	elif aID in [2, 3, 10, 11]:
		return np.pi / 2
	elif aID in [4, 5, 12, 13]:
		return np.pi
	else:
		return 0

#convert world state, actions to tensor
# FORWARD = (0, 5000)
# LEFT = (0.42, -50)
# RIGHT = (-0.9, -170)
# STOP = (0, -1e7)
# SIGNAL = (0, 0)

FORWARD = (0, 4000)
LEFT = (0.5, 0)
RIGHT = (-1.15, -1100)
STOP = (0, -1e7)
SIGNAL = (0, 0)

actionStringDict = {FORWARD:"forward", LEFT:"left", RIGHT:"right", STOP:"stop", SIGNAL:"signal"}
actionIntDict = {FORWARD:0, LEFT:1, RIGHT:2, STOP:3, SIGNAL:4}
def action_to_string(action):
	return actionStringDict[action.value]


def action_to_one_hot(action):
	action_space = ['forward', 'left', 'right', 'stop', 'signal']
	res = [0] * len(action_space)
	res[action_space.index(action)] = 1
	return res

def one_hot_to_action(one_hot_action):
	action_space = ['forward', 'left', 'right', 'stop', 'signal']
	actionString = action_space[np.argmax(one_hot_action)]
	if actionString == "forward":
		return Action.FORWARD
	elif actionString == "left":
		return Action.LEFT
	elif actionString == "right":
		return Action.RIGHT
	else:
		return Action.STOP
def state_action_to_joint_tensor(state, actions, device='cpu'):
	'''
	state should be a World object
	actions should be a dictionary of agentID --> action taken as an Action object

	tensor is a [a1_exists, a1_x, a1_y, a1_action_one_hot | ... | an_exists, an_x, an_y, an_action_one_hot | time ]
	'''
	res = []
	num_possible_agents = 16
	num_possible_actions = len(actionStringDict)
	for i in range(num_possible_agents):
		agent = state.agent_exists(i)
		if agent is None:
			exists = 0
			x = 0 
			y = 0 
			heading = 0
			action = action_to_one_hot('forward')  # give an arbitrary action if an agent doesn't exist
		else:
			exists = 1
			x = agent.x 
			y = agent.y
			heading = agent.heading
			try:
				action = action_to_one_hot(action_to_string(actions[i]))
			except:
				#  give an arbitrary action if we believe an agent to exist but it doesn't actually
				action = action_to_one_hot('forward')  
		res.append(exists)
		res.append(x)
		res.append(y)
		res.append(heading)
		for a in action:  # add the one hot
			res.append(a)
	res.append(state.t)
	return torch.tensor(res, device=device)

# convert tensor to world state
def joint_sa_tensor_to_state_action(joint_tensor):
	# convert tensor to numpy array
	representation = joint_tensor.detach().cpu().numpy()

	# Init the scenario
	scenario = Scenario1()
	state = scenario.w
	state.reset()  # remove all dynamic agents but keep buildings

	state.t = representation[-1]  # we stored time
	
	num_possible_agents = 16
	num_possible_actions = len(actionStringDict)

	single_agent_info_size = 4 + num_possible_actions

	actions = {}

	for i in range(0, len(representation)-1, single_agent_info_size):  # jump in chunks, don't include final time value
		curr = i
		agentID = i // single_agent_info_size
		exists = bool(representation[curr])
		if not exists:
			continue
		curr += 1

		[x, y, heading] = representation[curr: curr+3]
		

		curr += 3

		one_hot = representation[curr:curr+num_possible_actions]
		action = one_hot_to_action(one_hot)
		actions[agentID] = action 

		if i in [0, 1, 2, 3, 8, 9, 10, 11]:
			color = "red"
		else:
			color = "blue"

		agent = Car(Point(x, y), heading, ID=agentID, color=color)
		state.add(agent)

		actions[i] = action
	return state, actions
