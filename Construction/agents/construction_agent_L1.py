import pdb
import pickle
import torch
import test_reasoning_about_L0
import envs.construction as construction
import random
import numpy as np
import agents
import scipy.special
import test_construction_desire_pred
from utils import construction_data
import itertools
import agents.construction_agent_L0 as construction_agent_L0
import test_reasoning_about_construction_L0

def block_pair_utilities_to_desire_int(colored_block_utilities, num_possible_block_pairs=3):
	# Make a tuple of utility ints with a fixed order based on ALL_BLOCK_PAIRS
	utilities = []
	for block_pair in construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs]:
		try:
			utilities.append(colored_block_utilities[block_pair])
		except:
			pdb.set_trace()

	try:
		res = utilities.index(100)
	except:
		pdb.set_trace()
	return res

def desire_int_to_utilities(desire_int, num_possible_block_pairs):
	utilities = [0] * num_possible_block_pairs
	utilities[desire_int] = 100
	return dict(zip(construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities))







class AgentL1Random:
	def __init__(self):
		self.observations = []
		self.actions = []

	def get_action(self, observation=None):
		if observation is not None:
			self.observations.append(observation)

		action = random.choice(list(construction.Action))
		self.actions.append(action)
		return action

class AgentL1:
	"""Level-1 agent

	Args
		seek_conflict (bool)
		base_colored_block_utilities (only used if seek_conflict is False)
		inference_algorithm (str): one of ["NN", "SMC", "SMC+NN", "IS", "IS+NN",
										   "SMC+NN+rejuvenation", "oracle", "baseline",
										   "SMC(100)","Online_IS+NN"]
	"""
	def __init__(
		self,
		seek_conflict,
		base_colored_block_utilities,
		num_possible_block_pairs,
		initial_state_L0,
		initial_agent_location_L1,
		transition_L0,
		transition_L1,
		inference_algorithm,
		initial_block_picked=None,
		beta_L0=10.0,
		beta_L1=0.2,
		num_samples=100,
		model=None,
		ground_truth_colored_block_utilities_L0=None,
		visitedParticles={},
		useBFS=False
	):
		self.actions = []  # L1 actions
		self.seek_conflict = seek_conflict
		self.base_colored_block_utilities = base_colored_block_utilities
		self.num_possible_block_pairs = num_possible_block_pairs
		self.initial_state_L0 = initial_state_L0
		self.num_colored_block_locations = len(self.initial_state_L0.colored_blocks)
		self.observations = []  # L1 observations
		self.initial_agent_location_L1 = initial_agent_location_L1
		self.initial_agent_inv_L1 = initial_block_picked
		self.curr_state_L0 = initial_state_L0
		self.curr_state_L1 = construction.StateL1(
			colored_block_utilities_L0=construction.get_default_colored_block_utilities(
				self.num_possible_block_pairs
			),
			belief_L0=[(initial_state_L0, 1)],
			observation_L0=initial_state_L0,
			agent_location_L1=self.initial_agent_location_L1,
			agent_inv_L1=self.initial_agent_inv_L1
		)
		self.ground_truth_colored_block_utilities_L0 = ground_truth_colored_block_utilities_L0
		self.env_L0 = construction.ConstructionEnv(
			initial_state=initial_state_L0,
			colored_block_utilities=self.ground_truth_colored_block_utilities_L0
		)
		self.useBFS = useBFS
		# self.env_L0 = construction.ConstructionEnv(
		#     initial_state=initial_state_L0,
		#     colored_block_utilities=self.ground_truth_colored_block_utilities_L0
		# )

		self.transition_L0 = transition_L0
		self.transition_L1 = transition_L1
		if inference_algorithm in [
			"NN",
			"SMC",
			"SMC+NN",
			"IS",
			"IS+NN",
			"SMC+NN+rejuvenation",
			"oracle",
			"baseline",
			"SMC(100)",
			"Online_IS+NN",
			"random"
		]:
			self.inference_algorithm = inference_algorithm
		else:
			raise ValueError(
				"inference_algorithm must be one of ['NN', 'SMC', 'SMC+NN', 'IS', 'IS+NN',"
				"'SMC+NN+rejuvenation', 'oracle', 'baseline', 'SMC(100)', 'Online_IS+NN']"
			)
		self.beta_L0 = beta_L0
		self.beta_L1 = beta_L1

		self.num_samples = num_samples
		self.model = model
		self.particles = None
		self.log_weight = None
		self.L1_actions = []

		# mapping of util int to beliefs, env_clone, agent_clone, log weight, timestep to update from (len(states))
		self.prev_sampled_utilities = visitedParticles


		# Override inference algorithm and number of particles if it's SMC(100)
		if inference_algorithm == "SMC(100)":
			self.inference_algorithm = "SMC"
			self.num_samples = 100

	@property
	def num_timesteps(self):
		return len(self.observations)

	@property
	def states_L0(self):
		return [obs.state_L0 for obs in self.observations]

	@property
	def actions_L0(self):
		return [obs.action_L0 for obs in self.observations]

	def get_action_probs(self, agent_location_L1, belief=None, L2_reasoning=False, useBFS=False):
		"""Policy of the L1 agent based on observations so far.
		Math described on page 6 of
		https://drive.google.com/file/d/1KL89EEFNu3rt6e7AQ5FcoDSs255-1ZIu/view?usp=sharing

		Args
			agent_location_L1 (tuple of ints): location of the L1 agent
			belief (list of length up to num_samples): the length is variable because there could
				have been duplicate elements in the samples
					belief[i] = (food_truck_utilities_tuple, prob)
						food_truck_utilities_tuple = [(food_truck_name, utility), ...]
						prob = corresponding posterior probability

		Returns
			action_probs (list of length len(envs.food_trucks.Action)):
		"""
		if belief is None:
			belief = self.get_belief()

		action_space = list(construction.Action)
		uniform_probs = np.ones((len(action_space),)) / len(action_space)
		if belief is None:
			return uniform_probs
		else:
			state = construction.State(
				self.curr_state_L0.gridworld,
				agent_location_L1,
				self.curr_state_L0.colored_blocks,
				self.curr_state_L1.agent_inv_L1
			)
			# pdb.set_trace()
			expected_utility = {a: 0 for a in action_space}
			# values = {a: [] for a in action_space}

			if self.base_colored_block_utilities is not None:  # if L1 has a utility, search for that
				max_block = max(self.base_colored_block_utilities, key=self.base_colored_block_utilities.get)
				actions, values = construction_agent_L0.determine_subgoals(state_0=self.curr_state_L0,
																		   transition=self.transition_L0,
																		   colored_block_utilities=self.base_colored_block_utilities,
																		   state_1=state,
																		   agent_id=1,
																		   seek_conflict=False,
																		   useBFS=useBFS)
				for i in range(len(actions)):
					action = actions[i]
					value = values[i]
					expected_utility[construction.Action(action.value)] += value  # want to minimize value for L1 actions

			else:  # otherwise we are helping/hurting
				for (colored_block_utilities_L0, prob) in belief:
					L0_utils = dict(colored_block_utilities_L0)
					max_block = max(L0_utils, key=L0_utils.get)
					actions, values = construction_agent_L0.determine_subgoals(state_0=self.curr_state_L0,
																			   transition=self.transition_L0,
																			   colored_block_utilities=L0_utils,
																			   state_1=state,
																			   agent_id=1,
																			   seek_conflict=self.seek_conflict,
																			   useBFS=useBFS)
					# print(f"L1's Policy for the max pair being {max_block} when L1 hurting = {self.seek_conflict} is \n {actions} with a corresponding value of {values}")
					for i in range(len(actions)):
						action = actions[i]
						value = values[i]
						expected_utility[construction.Action(action.value)] += prob * value  # want to minimize value for L1 actions
				# print("---------")
				# values[action].append({"prob": prob, "val": value})



			# iterate through all possible utilities and probabilities for L0
				# assuming that utility is the ground truth, what is the best action and value
				# given the best value, multiply by probability of L0 utility
				# aggregate for each action




			# print(expected_utility)
			# print("values")
			# print(values)
			# print("policy")
			# print(policy)
			# pdb.set_trace()
			action_log_probs = np.full((len(action_space),), -1e6)
			for action_id, action in enumerate(action_space):
				action_log_probs[action_id] = self.beta_L1 * expected_utility[action]
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

	def get_nn_probs(self):
		"""[num_timesteps, num_possible_rankings]"""
		# Running a pure neural net
		# TODO make this online

		tensor_list = [
					construction_data.state_to_state_tensor(
						state, self.num_colored_block_locations, self.model.device
					)
					for state in self.states_L0
				]
		# for i, s in enumerate(self.states_L0):
		# 	if i == len(self.states_L0) - 1:
		# 		print(s)
		state_tensors = torch.stack(
				tensor_list,
				dim=0,
			)
		
		return test_construction_desire_pred.get_nn_probs(
			self.model,
			state_tensors,
			torch.stack(
				[
					construction_data.action_to_action_tensor(action, self.model.device)
					for action in self.actions_L0
				],
				dim=0,
			),
		)

	def get_belief(self):
		"""Computes a belief given all observations and actions up until now.
		A belief is the probability distribution p(colored_block_utilities_L0 | observations, action).

		Returns:
			belief (list of length up to num_samples): the length is variable because there could
			have been duplicate elements in the samples
				belief[i] = (colored_block_utilities_tuple, prob)
					colored_block_utilities_tuple = (block_pair, utility)
					prob = corresponding posterior probability
		"""

		if self.inference_algorithm == "NN" or self.inference_algorithm == "random":
			# Create an ordered list of colored_block_utilities
			# TODO: merge this with the snippet in test_desire_pred.get_prob_from_inference_output
			final_util_perms = []
			for i in range(self.num_possible_block_pairs):
				util = [0] * self.num_possible_block_pairs
				util[i] = 100
				final_util_perms.append(util) 
			colored_block_utilitiess = [
				tuple(sorted(zip(construction.ALL_BLOCK_PAIRS[:self.num_possible_block_pairs], utilities,)))
				for utilities in final_util_perms
			]

			# Combine with probs to make a belief
			if self.inference_algorithm == "random":  # just sample a single utility at random
				util = random.sample(colored_block_utilitiess, self.num_samples)
				return list(zip(util, [1/ len(colored_block_utilitiess)] * len(util)))

			return list(zip(colored_block_utilitiess, self.get_nn_probs()[-1]))
		elif self.inference_algorithm == "baseline":
			# Create an ordered list of colored_block_utilities
			# TODO: merge this with the snippet in test_desire_pred.get_prob_from_inference_output
			final_util_perms = []
			for i in range(self.num_possible_block_pairs):
				util = [0] * self.num_possible_block_pairs
				util[i] = 100
				final_util_perms.append(util) 
			colored_block_utilitiess = [
				tuple(sorted(zip(construction.ALL_BLOCK_PAIRS[:self.num_possible_block_pairs], utilities,)))
				for utilities in final_util_perms
			]

			probs = [1 / len(colored_block_utilitiess)] * len(colored_block_utilitiess)
			return list(zip(colored_block_utilitiess, probs))
		elif self.inference_algorithm == "oracle":
			if self.ground_truth_colored_block_utilities_L0 is None:
				raise RuntimeError("ground truth food truck utilities not set in oracle mode")
			return [(self.ground_truth_colored_block_utilities_L0, 1.0)]
		else:
			# Return current belief if we have already seen enough timesteps
			# if self.particles is not None and len(self.particles[1][0]) == self.num_timesteps:
			# 	colored_block_utilitiess, beliefss, env_clones, agent_clones = self.particles
			# 	return test_reasoning_about_construction_L0.get_posterior(
			# 		list(zip(colored_block_utilitiess, beliefss)), self.log_weight
			# 	)

			if self.inference_algorithm == "Online_IS+NN":
				colored_block_utilities_proposal_probs = self.get_nn_probs()[-1]
				# print(f"Raw NN output from agent {self.seek_conflict}: {colored_block_utilities_proposal_probs}")


				if self.num_timesteps == 1 and len(self.prev_sampled_utilities) == 0:
					visitedParticles = None 
				else:
					visitedParticles = self.prev_sampled_utilities

				self.particles, self.log_weight = test_reasoning_about_construction_L0.particle_inference(
					pickle.loads(pickle.dumps(self.env_L0)),
					pickle.loads(pickle.dumps(self.states_L0)),
					pickle.loads(pickle.dumps(self.actions_L0)),
					False,
					False,
					self.num_samples,
					[colored_block_utilities_proposal_probs],
					output_every_timestep=False,
					visitedParticles = visitedParticles
				)
			else:
				if self.num_timesteps == 1:
					assert self.particles is None

					if self.inference_algorithm == "SMC":
						colored_block_utilities_proposal_probs = None
					elif self.inference_algorithm == "SMC+NN":
						colored_block_utilities_proposal_probs = self.get_nn_probs()[-1]
					elif self.inference_algorithm == "SMC+NN+rejuvenation":
						colored_block_utilities_proposal_probs = self.get_nn_probs()[-1]
					elif self.inference_algorithm == "IS":
						colored_block_utilities_proposal_probs = None
					elif self.inference_algorithm == "IS+NN":
						colored_block_utilities_proposal_probs = self.get_nn_probs()[-1]
					else:
						raise NotImplementedError(f"{self.inference_algorithm} not implemented yet")

					# if self.inference_algorithm == 'SMC':
					#     pdb.set_trace()
					self.particles, self.log_weight = test_reasoning_about_construction_L0.init_particles(
						self.env_L0,
						data=(self.initial_state_L0, self.actions_L0[-1]),
						num_samples=self.num_samples,
						colored_block_utilities_proposal_probs=colored_block_utilities_proposal_probs,
						beta=self.beta_L0,
					)
				else:
					assert len(self.particles[1][0]) == self.num_timesteps - 1

					if self.inference_algorithm == "SMC":
						resample = True
						rejuvenate = False
						colored_block_utilities_proposal_probs = None
					elif self.inference_algorithm == "SMC+NN":
						resample = True
						rejuvenate = False
						colored_block_utilities_proposal_probs = None
					elif self.inference_algorithm == "SMC+NN+rejuvenation":
						resample = True
						rejuvenate = True
						colored_block_utilities_proposal_probs = self.get_nn_probs()[-1]
					elif self.inference_algorithm == "IS":
						resample = False
						rejuvenate = False
						colored_block_utilities_proposal_probs = None
					elif self.inference_algorithm == "IS+NN":
						resample = False
						rejuvenate = False
						colored_block_utilities_proposal_probs = self.get_nn_probs()[-1]
					else:
						raise NotImplementedError(f"{self.inference_algorithm} not implemented yet")

					# pdb.set_trace()
					# if self.states_L0[-1].block_picked is not None:
					#     pdb.set_trace()
					self.particles, self.log_weight = test_reasoning_about_construction_L0.update_particles(
						self.particles,
						self.log_weight,
						data=(
							self.states_L0[-2],
							self.actions_L0[-2],
							self.states_L0[-1],
							self.actions_L0[-1],
						),
						resample=resample,
						rejuvenate=rejuvenate,
						colored_block_utilities_proposal_probs=colored_block_utilities_proposal_probs,
					)
			colored_block_utilitiess, beliefss, env_clones, agent_clones = self.particles


			'''
			beliefss = [(obs_L0, 1.0), (obs_L0, 1.0), ... (obs_L0, 1.0)] - t timesteps
			self.particles[1] = beliefss
			self.particles[1][0] = (obs_L0, 1.0)
			len(self.particles[1][0]) = 2
			
			'''


			# store sampled utils so that we don't repeat computation
			# we only specifically do this at time 0 because every other time we have it done in particle inference
			if self.num_timesteps == 1:
				for i, util in enumerate(colored_block_utilitiess):
					util_int = block_pair_utilities_to_desire_int(util, self.num_possible_block_pairs)
					infoToStore = (beliefss[i], env_clones[i], agent_clones[i], self.log_weight[i])
					self.prev_sampled_utilities[(util_int,1)] = infoToStore


			return test_reasoning_about_construction_L0.get_posterior(
				list(zip(colored_block_utilitiess, beliefss)), self.log_weight
			)
	def get_action(self, observation=None, return_info=False):
		if observation is not None:
			assert observation not in self.observations
			self.observations.append(observation)

		if self.num_timesteps == 0:
			if return_info:
				return construction.Action.STOP, None
			else:
				return construction.Action.STOP
		else:
			belief = self.get_belief()
			action_probs = self.get_action_probs(self.curr_state_L1.agent_location_L1, belief=belief, useBFS=self.useBFS)
			action_space = list(construction.Action)
			action_to_prob = dict(zip(action_space, action_probs))
			#print(f"L1 action to probabilities: {action_to_prob}")

			# action = np.random.choice(action_space, p=action_probs)  # TODO: Uncomment for stochastic action

			max_actions = []
			max_action_prob = np.max(action_probs)
			for i, a in enumerate(action_space):
				if action_probs[i] == max_action_prob:
					max_actions.append(a)
			action = max_actions[np.random.randint(0, len(max_actions))]  # FOR DETERMINISTIC L1

			# self.agent_location_L1 = self.transition_L1(self.agent_location_L1, action)

			if return_info:
				return action, {"belief": belief, "action_probs": action_probs, "conflict": self.seek_conflict,
								"prev_L1_action": self.L1_actions[-1]}
			else:
				return action
