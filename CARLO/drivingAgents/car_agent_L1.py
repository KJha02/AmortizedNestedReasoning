import scipy.special
import _pickle as pickle
from enum import Enum
from collections import deque
import itertools
import numpy as np
from scipy.special import softmax
import torch
import random
from randomStreetGen import isVisible
from agents import Car, RectangleBuilding, Pedestrian, Painting
from scenario import Action, get_partial_states
from drivingAgents.car_agent_L0 import AgentL0
from geometry import Point, Line
import pdb
import reasoning_about_car_L0
import drivingAgents.car_agent_L0 as car_agent_L0
from car_utils.network import agent_pair_to_tensor, state_action_to_joint_tensor

class Action(Enum):
	FORWARD = (0, 4000)
	LEFT = (0.5, 0)
	RIGHT = (-1.15, -1100)
	STOP = (0, -1e7)
	SIGNAL = (0, 0)

# class Action(Enum):
# 	FORWARD = (0, 5000)
# 	LEFT = (0.42, -50)
# 	RIGHT = (-0.9, -170)
# 	STOP = (0, -1e7)
# 	SIGNAL = (0, 0)


class AgentL1:
	"""Level-1 agent

	Args personal goal string ("forward", 'left', or 'right')
		personal goal mapping - Point(x, y) of target, used for evaluation of action quality
		initialLocMapping - dict of id to initial loc
		inference algorithm (IS, NN, IS+NN)
		car_exists_prior - what percentage do we give to other cars for thinking a certain missing car exists
		signalDangerWeight - how to balance own goal versus stopping to signal 

	"""
	def __init__(
		self,
		agentID,
		goalActionString,
		initialLocMapping,
		scenario,
		transition,
		inference_algorithm,
		signalDangerWeight=0.5,
		car_exists_prior = 0.5,
		beta_L0=0.001,
		beta_L1=0.001,
		num_samples=3,
		lookAheadDepth=10,
		model=None,
		exist_model = None,
		state_model = None,
		visitedParticles={}
	):

		self.scenario = scenario

		self.personal_L0 = AgentL0(agentID, goalActionString, transition, initialLocMapping, carExistPrior=car_exists_prior, beta=beta_L1,
			lookAheadDepth=lookAheadDepth, state_model=state_model, exist_model=exist_model, inference_algorithm=inference_algorithm)

		self.agentID = agentID

		self.transition = transition


		# maps carID (int) to fixed potential initial locations on map Point(x, y)
		self.initialLocMapping = initialLocMapping
		self.num_possible_cars = len(initialLocMapping)

		self.goalActionString = goalActionString
		self.goalAction, self.targetSlot = car_agent_L0.goalToLoc(self.agentID, self.goalActionString, self.initialLocMapping)  # high level action goal (really it's a Point(x, y) target location)
		self.beta_L0 = beta_L0
		self.beta_L1 = beta_L1

		self.lookAheadDepth = lookAheadDepth

		# probability for an unobserved car to exist
		self.carExistPrior = car_exists_prior

		self.num_samples = num_samples
		self.signalDangerWeight = signalDangerWeight

		self.model = model  # goal inference model
		self.state_model = state_model
		self.exist_model = exist_model

		if inference_algorithm in [
			"NN",
			"IS",
			"IS+NN",
			"Online_IS+NN",
			"random"
		]:
			self.inference_algorithm = inference_algorithm

		self.idToParticlesWeight = {}  # goes from id to a dictionary mapping particles to a list and log weight to a list

		self.personal_actions = []

		self.particles = {}
		self.log_weight_dict = {}
		self.action_weight_dict = {}
		for i in range(16):
			self.particles[i] = None
			self.log_weight_dict[i] = None
			self.action_weight_dict[i] = None
 
		self.previous_sampled = visitedParticles  # maps agent id to dictionary mapping (goal, time) to particle

		# partial observations from the world, each one should be a CARLO world object
		self.partial_obs = []  # contains tuple of (World, dictionary mapping id to actions taken from that world)


	@property
	def num_timesteps(self):
		return len(self.partial_obs)

	@property
	def visible_states(self):
		return [x[0] for x in self.partial_obs]

	def visible_actions(self, agentID):  # needs to be action history for agent
		return [x[1][agentID] for x in self.partial_obs]

	def first_visible(self, states, targetID):
		for t, s in enumerate(states):
			for car in s.dynamic_agents:
				if car.ID == targetID:
					return t
		return None




	def get_nn_probs(self, target_agent):
		"""[num_timesteps, num_possible_goals]"""
		# Running a pure neural net to reason about target agent intentions
		# TODO make this online

		pair_tensor = agent_pair_to_tensor(self.agentID, target_agent, device=self.model.device)
		state_actions = []
		for i, partial_state in enumerate(self.partial_obs):
			sa_tensor = state_action_to_joint_tensor(partial_state[0], partial_state[1], device=self.model.device)
			state_actions.append(sa_tensor)
		nn_input = torch.stack(state_actions, dim=0)
		lens = torch.LongTensor([len(nn_input)]).cpu()
		log_prob = self.model([nn_input], [pair_tensor], lens)
		probs = torch.softmax(log_prob, 1).cpu().detach().numpy()
		probs /= probs.sum(axis=1, keepdims=True)
		return probs


	def get_belief(self, target_agent):
		'''
		ASSUME THIS ONLY GETS CALLED FOR AGENTS YOU CAN SEE
		Returns current agent's inference about another agent
		'''
		visibleStates = self.visible_states 
		visibleActions = self.visible_actions(target_agent)
		startTime = self.first_visible(visibleStates, target_agent)
		fullActions = [x[1] for x in self.partial_obs][startTime:]

		if self.inference_algorithm == "NN":
			possible_goals = ['forward', 'left', 'right']

			return list(zip(possible_goals, self.get_nn_probs(target_agent)[-1]))

		elif self.inference_algorithm == "Online_IS+NN" or self.inference_algorithm == "random":
			prior = self.get_nn_probs(target_agent)
			if self.inference_algorithm == "random":
				prior = np.full_like(prior, 1/3)
			if self.num_timesteps == 1 and len(self.previous_sampled[target_agent]) == 0:
				visitedParticles = None 
			else:
				visitedParticles = self.previous_sampled[target_agent]

			self.particles[target_agent], self.log_weight_dict[target_agent], self.action_weight_dict[target_agent] = reasoning_about_car_L0.particle_inference(
				target_agent, 
				self.scenario, 
				visibleStates[startTime:],
				visibleActions[startTime:], 
				num_samples=self.num_samples,
				lane_utilities_proposal_probss=prior,
				output_every_timestep=False,
				carExistPrior=self.carExistPrior,
				lookAheadDepth=self.lookAheadDepth,
				visitedParticles=visitedParticles,
				beta=self.beta_L0,
				full_action_history=fullActions,
				state_belief_model=self.state_model, 
				exist_belief_model=self.exist_model,
				inference_algorithm=self.inference_algorithm
			)
		else:  # Importance sampling
			if self.num_timesteps == startTime + 1 and len(self.previous_sampled[target_agent]) == 0:
				visitedParticles = None 
			else:
				visitedParticles = self.previous_sampled[target_agent]
			
			if self.num_timesteps == startTime + 1:
				if self.inference_algorithm == "IS+NN":
					nn_prior = self.get_nn_probs()[-1]
				elif self.inference_algorithm == "IS":
					nn_prior = None

				self.particles[target_agent], self.log_weight_dict[target_agent], next_action_probs = reasoning_about_car_L0.init_particles(
					self.scenario,
					data=(visibleStates[startTime], visibleActions[startTime], target_agent),
					num_samples=self.num_samples,
					history = visitedParticles,
					lane_utilities_proposal_probs=nn_prior,
					carExistPrior=self.carExistPrior,
					lookAheadDepth=self.lookAheadDepth,
					beta=self.beta_L0
				)
				self.action_weight_dict[target_agent]= {0: next_action_probs} 
				
			else:
				timestep = self.num_timesteps - 1
				self.particles[target_agent], self.log_weight_dict[target_agent], next_action_probs = reasoning_about_car_L0.update_particles(
					particles=self.particles[target_agent],
					log_weight=self.log_weight_dict[target_agent],
					data=(visibleStates[timestep - 1], visibleActions[timestep - 1], visibleStates[timestep], visibleActions[timestep], target_agent),
					lane_utilities_proposal_probs=None,
				)
				self.action_weight_dict[target_agent][self.num_timesteps - 1] = next_action_probs



		# store sampled utils so that we don't repeat computation
		# we only specifically do this at time 0 because every other time we have it done in particle inference
		(lane_utilitiess, beliefss, agent_clones) = self.particles[target_agent]
		next_action_weights = self.action_weight_dict[target_agent]
		shiftedTime = max(next_action_weights) + 1
		log_weights = self.log_weight_dict[target_agent]
		for i, util in enumerate(lane_utilitiess):
			if self.previous_sampled[target_agent] is None:
				self.previous_sampled[target_agent] = {}
			if (util, shiftedTime) not in self.previous_sampled[target_agent]:
				try:
					self.previous_sampled[target_agent][(util, shiftedTime)] = (beliefss[i], agent_clones[i], log_weights[i], next_action_weights[shiftedTime-1][util])
				except:
					pdb.set_trace()

		return reasoning_about_car_L0.get_posterior(
			list(zip(lane_utilitiess, beliefss)), log_weights
		)

	def get_action_probs(self, beliefDict=None):
		recent_state_observation, recent_action_observation = self.partial_obs[-1]

		observableAgentIDs = []

		for otherCar in recent_state_observation.dynamic_agents:
			if otherCar != self.agentID:
				observableAgentIDs.append(otherCar.ID)


		if beliefDict is None:  # get our goal inference if we don't already have it
			beliefDict = {}
			for otherCar in recent_state_observation.dynamic_agents:
				if otherCar.ID != self.agentID:
					# dict of (lane utility -> prob) for each agent from curr perspective
					beliefDict[otherCar.ID] = dict(self.get_belief(otherCar.ID))

		action_space = list(Action)
		uniform_probs = np.ones((len(action_space),)) / len(action_space)

		if beliefDict is None:
			return uniform_probs

		expected_utility = {a: None for a in action_space}
		# Begin by just determining the best actions contingent on 
		L1_state_belief = self.personal_L0.get_belief()[0]
		actions, values = car_agent_L0.determine_subgoals(L1_state_belief, self.transition, self.agentID, self.goalAction, self.goalActionString, self.lookAheadDepth)
		for i, action in enumerate(actions):
			expected_utility[Action(action.value)] = values[i]

		# iterate through every state you believe in, look into the future to determine other crashing prob
		crashProb = 0.0
		for state, prob in L1_state_belief:
			curr_state = pickle.loads(pickle.dumps(state))
			curr_action_dict = pickle.loads(pickle.dumps(recent_action_observation))
			for t in range(self.lookAheadDepth):
				temp_action_dict = {}
				for car in curr_state.dynamic_agents: 
					if car.ID == self.agentID:
						temp_action_dict[car.ID] = self.personal_L0.get_action(curr_state, prev_actions=curr_action_dict)
						self.personal_L0.observations.pop()  # don't want to actually keep mem
						self.personal_L0.full_action_history.pop()
					elif car.ID not in observableAgentIDs or not curr_state.agent_exists(car.ID):  # one we imagine to exist, so it keeps prev action
						temp_action_dict[car.ID] = car.prev_action
					elif self.inference_algorithm == 'random':
						temp_action_dict[car.ID] = random.choice(action_space[:4])
					else:
						intentionBelief = beliefDict[car.ID]  # what do we belief about its goal
						action_probabilities = [1e-6] * 5  # L0's only have 4 actions with the 5th as a dummy
						L0_agent_clones = self.particles[car.ID][2]

						for agent_clone in L0_agent_clones:
							goal = agent_clone.goalActionString
							goal_prob = intentionBelief[goal]
							L0_potential_action, L0_potential_action_info = agent_clone.get_action(curr_state, return_info=True, prev_actions=curr_action_dict)  # get action probs
							L0_potential_action_probs = L0_potential_action_info['action_probs']
							for i, a_prob in enumerate(L0_potential_action_probs):  # weigh overall L0 action by action probs
								action_probabilities[i] += goal_prob * a_prob
							agent_clone.observations.pop()  # remove curr_state from imagined future so we don't impact the gt
							agent_clone.full_action_history.pop()
						log_probs = np.log(action_probabilities)

						normalized_log_probs = log_probs - scipy.special.logsumexp(log_probs)
						action_probabilities = list(softmax(normalized_log_probs))
						
						max_actions = [a for i, a in enumerate(action_space[:4]) if action_probabilities[i] == max(action_probabilities)]
						random.shuffle(max_actions)  # avoiding uniform being stuck
						temp_action_dict[car.ID] = random.choice(max_actions)


				next_state = self.transition(curr_state, temp_action_dict)  # step into future using reactive actions

				if next_state.collision_exists():  # if we anticipate anyone crashing, we signal
					# crashProb += ((0.99 ** t) * prob)
					crashProb += prob
					break

				curr_state = next_state  # make sure to step through
				curr_action_dict = pickle.loads(pickle.dumps(temp_action_dict))


		signalProb = (crashProb * self.signalDangerWeight)  # probability of someone crashing and you reporting it
		ownGoalProb = 1 - signalProb

		# signalOnly = random.random() <= signalProb
		signalOnly = crashProb > 0
		
		action_log_probs = np.full((len(action_space),), -1e6)
		for action_id, action in enumerate(action_space):
			if signalOnly and action.value != Action.SIGNAL.value:
				action_log_probs[action_id] = -1e6
			elif signalOnly and action.value == Action.SIGNAL.value:
				action_log_probs[action_id] = self.beta_L1 * 100 * signalProb
			elif expected_utility[action] is not None:
				action_log_probs[action_id] = self.beta_L1 * expected_utility[action] * ownGoalProb
			# if action.value == Action.SIGNAL.value:
			# 	action_log_probs[action_id] = self.beta_L1 * 10 * signalProb

		if scipy.special.logsumexp(action_log_probs) < np.log(1e-6):
			return uniform_probs, beliefDict
		else:
			action_log_probs_normalized = action_log_probs - scipy.special.logsumexp(
				action_log_probs
			)
			action_probs_normalized = np.exp(action_log_probs_normalized)
			action_probs_normalized = action_probs_normalized / np.sum(action_probs_normalized)
			if np.isnan(action_probs_normalized).any():
				raise RuntimeError("nan action probs")
			return action_probs_normalized, beliefDict


	def get_action(self, observation=None, return_info=False):
		if observation is not None:
			self.partial_obs.append(observation)
			self.personal_L0.observations.append(observation[0])
			self.personal_L0.full_action_history.append(observation[1])

		action_probs, beliefDict = self.get_action_probs()
		action_space = list(Action)
		max_actions = [a for i, a in enumerate(action_space) if action_probs[i] == max(action_probs)]
		random.shuffle(max_actions)
		action = np.random.choice(max_actions)  # pick a random maximizing action
		# action = np.random.choice(action_space, p=action_probs)
		if self.goalActionString != "right" and action == Action.RIGHT:
			action_probs = self.get_action_probs(belief=beliefDict)
			
		if return_info:
			return action, {"belief": beliefDict, "action_probs": action_probs}
		else:
			return action


