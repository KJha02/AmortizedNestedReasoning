"""
Implements importance sampling described in
https://drive.google.com/file/d/129VU_YM4plV_rtSPYX6AJrIuZNLiNtod/view?usp=sharing
"""

import copy
import numpy as np
import test_sample_construction_L0_rollout
import agents.construction_agent_L0
import envs.construction_sample
import pdb
import utils.construction_data
import tqdm
import utils.general
import scipy.special
import envs
import envs.construction
import _pickle as pickle
import agents.construction_agent_L0 as construction_agent_L0

def translate_util(utilities):
	max_pair = max(utilities, key=utilities.get)
	return (envs.construction.block2color[max_pair[0]], envs.construction.block2color[max_pair[1]])

def block_pair_utilities_to_desire_int(colored_block_utilities, num_possible_block_pairs=3):
	# Make a tuple of utility ints with a fixed order based on ALL_BLOCK_PAIRS
	utilities = []
	for block_pair in envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs]:
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
	return dict(zip(envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities))


def get_corrected_log_weight(log_weights):
	"""
	Weight correction.

	Set weights to uniform if they are too small because that means that
	the action probs are too small which means that that action is not
	in any shortest path.
	Not doing this can lead to non-sensical L2 planning which relies on
	the log_weight for ALL actions some of which are bound to not be in the
	shortest path.

	Args
		log_weights (np.array [num_samples])

	Returns
		corrected_log_weights (np.array [num_samples])
	"""
	log_weight_threshold = float("-inf")  # TODO: Make adjustable
	if np.max(log_weights) < log_weight_threshold:
		print("WARNING: log_weights are too small")
		# pdb.set_trace()
		return np.zeros_like(log_weights), True
	else:
		return log_weights, False

def init_particles(env, data, num_samples, colored_block_utilities_proposal_probs=None, beta=0.01, history=None):
	"""
	Args
		env
		data
			state
			action
		num_samples (int)
		colored_block_utilities_proposal_probs (np.array or list of floats)
		beta

	Returns
		particles
			colored_block_utilitiess: List of length num_samples where each element is a dict
			beliefss: List of length num_samples where each element is list of length
				num_timesteps=1
			env_clones: list of length num_samples
			agent_clones: list of length num_samples
		log_weight [num_samples]
	"""
		# Extract
	state, action = data
	log_weight = np.zeros((num_samples,))

	# Make environment and agent clones (length num_samples)
	env_clones, agent_clones = [], []
	# List of length num_samples where each element is a dict
	colored_block_utilitiess = []
	# List of length num_samples where each element will eventually be a
	# list of length num_timesteps
	beliefss = [[] for _ in range(num_samples)]

	distinct_utils = []
	num_possible_block_pairs = len(env.colored_block_utilities)
	for util_id in range(num_possible_block_pairs):
		sample_util = utils.construction_data.desire_int_to_utilities(util_id, num_possible_block_pairs)
		distinct_utils.append(sample_util)
	# while len(distinct_utils) < num_possible_block_pairs:
	# 	sample_util = envs.construction_sample.sample_block_pair_utilities(num_possible_block_pairs, return_prob=False)
	# 	if sample_util not in distinct_utils:
	# 		distinct_utils.append(sample_util)

	#print(f"Action: {action}")
	topKUtils = []
	if colored_block_utilities_proposal_probs is not None:
		sortedColoredProbs = sorted(colored_block_utilities_proposal_probs, reverse=True)
		colored_block_utilities_proposal_probs = list(colored_block_utilities_proposal_probs)
		
		# for idx in range(len(colored_block_utilities_proposal_probs)):
		# 	colored_block_utility = pickle.loads(pickle.dumps(utils.construction_data.desire_int_to_utilities(idx, num_possible_block_pairs)))
		# 	max_pair = max(colored_block_utility, key=colored_block_utility.get)
		# 	probability = colored_block_utilities_proposal_probs[idx]
		# 	# print(f"ID: {idx}; Utility: {max_pair}; probability: {probability}")


		# print(f"regular {colored_block_utilities_proposal_probs}")
		# print(f"sorted {sortedColoredProbs}")
		# print("\n")
		# print(f"Colored block probs from trc0: {colored_block_utilities_proposal_probs}")
		for idx in range(num_samples):
			trueDesireInt = colored_block_utilities_proposal_probs.index(sortedColoredProbs[idx])
			# print(trueDesireInt)
			colored_block_utility = pickle.loads(pickle.dumps(utils.construction_data.desire_int_to_utilities(trueDesireInt, num_possible_block_pairs)))
			probability = sortedColoredProbs[idx]
			max_pair = max(colored_block_utility, key=colored_block_utility.get)
			# print(f"ID: {trueDesireInt}; Utility: {max_pair}; probability: {probability}")
			# # print(max_pair)
			# if max_pair == ('#', '%') and (num_samples == 5 or num_samples == 15):
			# 	print(f"GT NN prob = {sortedColoredProbs[idx]}, IDX is {idx}")
			# 	gt_idx = idx
			# 	pdb.set_trace()

			topKUtils.append((colored_block_utility, sortedColoredProbs[idx]))
		# print(sortedColoredProbs[:num_samples])
		# max_pair = max(topKUtils[0], key=topKUtils[0].get)
		# print(max_pair)
		# for top in topKUtils:
		# 	max_pair = max(top[0], key=top[0].get)
		# 	print(max_pair)

		# print("finished top K particles\n\n")
		# exit(0)

		# if num_samples == 15:
		# 	print(sortedColoredProbs)

	for sample_id in range(num_samples):
		# Sample θ ~ p(θ)
		if colored_block_utilities_proposal_probs is None:
			if sample_id < len(distinct_utils) and num_samples >= num_possible_block_pairs:
				colored_block_utilities = distinct_utils[sample_id] # TODO: uncomment for actual particle sampling
				# colored_block_utilities = env.colored_block_utilities # uncomment for no inference -> GT sampling
			else:  # if we have more samples than distinct utilities or we aren't fully sampling all particles
				colored_block_utilities = np.random.choice(distinct_utils)
		else:
			if sample_id < len(topKUtils):
				(colored_block_utilities, colored_block_utilities_proposal_prob) = topKUtils[sample_id]
			else:
				(
					colored_block_utilities,
					colored_block_utilities_proposal_prob,
				) = envs.construction_sample.sample_block_pair_utilities(
					num_possible_block_pairs,
					prior=colored_block_utilities_proposal_probs,
					return_prob=True,
				)

		colored_block_utilitiess.append(colored_block_utilities)

		try:  # don't recompute initial point if we're going to replace it anyways
			# since we're storing every (goal, time) combo in history, if the later times exist the first one exists
			if (utils.construction_data.block_pair_utilities_to_desire_int(colored_block_utilities, num_possible_block_pairs), 1) in history:
				beliefss.append(None)
				env_clones.append(None)
				agent_clones.append(None)
				continue
		except:
			pass


		env_clone = pickle.loads(pickle.dumps(env))
		env_clone.colored_block_utilities = pickle.loads(pickle.dumps(colored_block_utilities))
		env_clones.append(env_clone)
		agent_clones.append(
			agents.construction_agent_L0.AgentL0(
				env_clone.state.gridworld,
				env_clone.colored_block_utilities,
				env_clone.transition,
				beta,
			)
		)

		# PROPOSE
		# Sample o_1 ~ p(o | s_1)
		obs = env_clones[sample_id].reset()
		agent_clones[sample_id].observations.append(obs)

		# Sample b_1 ~ p(b | ...)
		belief = agent_clones[sample_id].get_belief()
		beliefss[sample_id].append(belief)

		# WEIGH based on p(a_1 | ...)p(s | ...)
		# Score p(a_1 | ...)
		action_prob = agent_clones[sample_id].get_action_probs(belief=belief)[action.value]

		# full_action_prob = agent_clones[sample_id].get_action_probs(belief=belief)

		#print(f"P(a | g) : {colored_block_utilities} -> {action_prob}")
		# print(dict(zip(list(agents.construction_agent_L0.Action), full_action_prob)))
		# print(full_action_prob)

		# Assign the weight

		if colored_block_utilities_proposal_probs is None or True:  # ignore the else statement for now
			if action_prob < 1e-6:
				log_weight[sample_id] = -1e6
			else:
				log_weight[sample_id] = np.log(action_prob)
		else:
			action_prob = max(1e-6, action_prob)
			log_weight[sample_id] = (
				np.log(action_prob)
				+ np.log(len(colored_block_utilities_proposal_probs))
				- np.log(colored_block_utilities_proposal_prob)
			)
	# print("-------\n")
	# pdb.set_trace()

	# if num_samples == 5 or num_samples == 15:
	# 	try:
	# 		justUtils = [x[0] for x in topKUtils]
	# 		assert colored_block_utilitiess == justUtils
	# 	except:
	# 		pdb.set_trace()

	# Weight correction
	log_weight, too_low = get_corrected_log_weight(log_weight)

	particles = colored_block_utilitiess, beliefss, env_clones, agent_clones
	return particles, log_weight

def update_particles(
	particles,
	log_weight,
	data,
	resample=True,
	rejuvenate=False,
	colored_block_utilities_proposal_probs=None,
):
	"""Go from (θ, b_{1:t - 1}) to (θ, b_{1:t}) with or without resampling.
	Used for importance sampling and SMC.

	Args
		particles
			colored_block_utilitiess: List of length num_samples where each element is a dict
			beliefss: List of length num_samples where each element is list of length num_timesteps
				(at least 1)
			env_clones: list of length num_samples
			agent_clones: list of length num_samples
		log_weight [num_samples]
		data
			prev_state
			prev_action
			state
			action
		resample (bool)
		rejuvenate (bool)
		colored_block_utilities_proposal_probs (None or np.array or list of floats)

	Returns
		updated_particles
			updated_colored_block_utilitiess: List of length num_samples where each element is a dict
			updated_beliefss: List of length num_samples where each element is list of length
				(num_timesteps + 1)
			updated_env_clones: list of length num_samples
			updated_agent_clones: list of length num_samples
		updated_log_weight [num_samples]
	"""
		# Extract
	num_samples = len(log_weight)
	colored_block_utilitiess, beliefss, env_clones, agent_clones = particles
	prev_state, prev_action, state, action = data
	updated_log_weight = np.zeros((num_samples,))
	num_timesteps = len(beliefss[0])

	for sample_id in range(num_samples):
		assert len(beliefss[sample_id]) == num_timesteps
	if resample:
		# RESAMPLE
		# Sample ancestral indices
		ancestral_index = utils.general.sample_ancestral_index(log_weight)

		# Reassign utilities
		colored_block_utilitiess_new = []
		for sample_id in range(num_samples):
			colored_block_utilitiess_new.append(colored_block_utilitiess[ancestral_index[sample_id]])
		colored_block_utilitiess = colored_block_utilitiess_new

		# Reassign beliefs
		beliefss_new = []
		for sample_id in range(num_samples):
			beliefss_new.append(pickle.loads(pickle.dumps(beliefss[ancestral_index[sample_id]])))
		beliefss = beliefss_new
		for sample_id in range(num_samples):
			assert len(beliefss[sample_id]) == num_timesteps

		# Reassign envs
		env_clones_new = []
		for sample_id in range(num_samples):
			env_clones_new.append(env_clones[ancestral_index[sample_id]])
		env_clones = env_clones_new

		# Reassign agents
		agent_clones_new = []
		for sample_id in range(num_samples):
			agent_clones_new.append(agent_clones[ancestral_index[sample_id]])
		agent_clones = agent_clones_new

	if rejuvenate and colored_block_utilities_proposal_probs is not None:
		# REJUVENATE θ based on q(θ | s_{1:t}, a_{1:t})
		# At the moment, we sample directly from q without any correction.
		num_possible_block_pairs = len(colored_block_utilitiess[0])
		for sample_id in range(num_samples):
			colored_block_utilities = envs.construction_sample.sample_block_pair_utilities(
				num_possible_block_pairs,
				prior=colored_block_utilities_proposal_probs,
				return_prob=False,
			)
			colored_block_utilitiess[sample_id] = colored_block_utilities

			env_clones[sample_id].colored_block_utilities = colored_block_utilitiess[sample_id]
			agent_clones[sample_id].colored_block_utilities = colored_block_utilitiess[sample_id]
	#print(f"Action: {action}")
	for sample_id in range(num_samples):
		# Step the environment
		env_clones[sample_id].state = state.clone()
		# if env_clones[sample_id].colored_block_utilities == {('=', 'x'): 100, ('=', '+'): 0, ('x', '+'): 0} and \
		#         state.block_picked is not None:
		#     pdb.set_trace()
		# PROPOSE
		# Sample o_t ~ p(o | ...)
		obs = env_clones[sample_id].get_observation()
		agent_clones[sample_id].observations.append(obs)

		# Sample b_t ~ p(b | ...)
		belief = agent_clones[sample_id].get_belief()
		beliefss[sample_id].append(belief)
		assert len(beliefss[sample_id]) == num_timesteps + 1

		# WEIGH based on p(a_t | ...)p(s_t | ...)
		# Score p(a_t | ...)
		# try:

		action_prob = agent_clones[sample_id].get_action_probs(belief=belief)[action.value]
		# except:
		#     pdb.set_trace()
		# Score p(s | ...)
		# print(f"P(a | g) : {colored_block_utilitiess[sample_id]} -> {action_prob}")

		# Assume state transition is deterministic so just need to make sure that the
		# next step is correct
		# try:
		#     assert env_clones[sample_id].transition(prev_state, action) == state
		# except:
		#     pdb.set_trace()

		# Assign the weight
		if action_prob < 1e-6:
			log_action_prob = -1e6
		else:
			log_action_prob = np.log(action_prob)
		if resample:
			updated_log_weight[sample_id] = log_action_prob
		else:
			alpha = 0.9
			p_uniform = 1 / num_samples
			updated_log_weight[sample_id] = np.log(alpha * np.exp(log_weight[sample_id]) + (1 - alpha) * p_uniform) + log_action_prob

		# Assert
		for i in range(sample_id):
			assert len(beliefss[i]) == num_timesteps + 1
		for i in range(sample_id + 1, num_samples):
			assert len(beliefss[i]) == num_timesteps

	# Assert
	for sample_id in range(num_samples):
		assert len(beliefss[sample_id]) == num_timesteps + 1
	updated_particles = colored_block_utilitiess, beliefss, env_clones, agent_clones
	# Weight correction
	corrected_log_weight, too_low = get_corrected_log_weight(updated_log_weight)

	# print(action)
	# for sample_id in range(num_samples):
	#
	#     belief = agent_clones[sample_id].get_belief()
	#     utility = colored_block_utilitiess[sample_id]
	#     action_prob = agent_clones[sample_id].get_action_probs(belief=belief)
	#     print(belief)
	#     print(utility)
	#     print(action_prob)

	# pdb.set_trace()

	return updated_particles, corrected_log_weight

def particle_inference(
	env,
	states,
	actions,
	resample,
	rejuvenate,
	num_samples=3,
	colored_block_utilities_proposal_probss=None,
	output_every_timestep=False,
	visitedParticles=None
):
	"""Particle-based inference either using SMC or importance sampling.

	Args
		env (envs.construction.ConstructionEnv)
		states (list of envs.construction.State of length num_timesteps)
		actions (list of envs.construction.Action of length num_timesteps)
		resample (bool): True = SMC, False = importance sampling
		rejuvenate (bool): True = rejuvenate θ, False = do not rejuvenate θ
		num_samples (int)
		colored_block_utilities_proposal_probss (np.ndarray [num_timesteps, num_possible_rankings] or
											  [num_timesteps, num_possible_block_pairs])
		output_every_timestep (bool): if True, output posterior at every timestep, otherwise output
			posterior at the last timestep

	Returns
		if output_every_timestep == False:
			particles: list of length num_samples where each element is a tuple containing
				colored_block_utilities (dict where keys are tuple of block pairs and values are utilities)
				beliefs (list of length num_timesteps)
			log_weights (np.ndarray of shape [num_samples])
		else:
			list of length num_timesteps where each element is the above tuple of (particles, log_weights)
	"""
	num_timesteps = len(states)
	num_possible_block_pairs = len(env.colored_block_utilities)
	# for i in range(len(states)):
	#     try:
	#         if i > 1:
	#             assert env.transition(states[i-1], actions[i-1]) == states[i]
	#     except:
	#         pdb.set_trace()
	# Initialize weights
	log_weights = np.zeros((num_samples, num_timesteps))
	# FIRST TIMESTEP
	if colored_block_utilities_proposal_probss is None:
		colored_block_utilities_proposal_probs = None
	else:
		if resample:
			if rejuvenate:
				colored_block_utilities_proposal_probs = colored_block_utilities_proposal_probss[0]
			else:
				colored_block_utilities_proposal_probs = colored_block_utilities_proposal_probss[0]
		else:
			colored_block_utilities_proposal_probs = colored_block_utilities_proposal_probss[0]
	(colored_block_utilitiess, beliefss, env_clones, agent_clones), log_weights[:, 0] = init_particles(
		env,
		data=(states[0], actions[0]),
		num_samples=num_samples,
		colored_block_utilities_proposal_probs=colored_block_utilities_proposal_probs,
		history=visitedParticles
	)
	if output_every_timestep:
		result = [(list(zip(colored_block_utilitiess, beliefss)), log_weights[:, 0])]


	if visitedParticles is not None:  # key is (integer of utility, start_time)
		for i, util in enumerate(colored_block_utilitiess):
			desire_int = block_pair_utilities_to_desire_int(util, num_possible_block_pairs)

			start_time = None
			for t in range(num_timesteps, 0, -1):
				if (desire_int, t) in visitedParticles:  # load in existing particle if it was sampled
					start_time = t
					belief, env_clone, agent_clone, log_weight = visitedParticles[(desire_int,t)]
					beliefss[i] = belief
					env_clones[i] = env_clone
					agent_clones[i] = agent_clone
					log_weights[i, start_time - 1] = log_weight
					break
			if start_time is None:  # otherwise we update from beginning
				start_time = 1

			for timestep in tqdm.tqdm(range(start_time, num_timesteps)):  # only update a single particle from where it left off
				if colored_block_utilities_proposal_probss is None:
					colored_block_utilities_proposal_probs = None
				else:
					if resample:
						if rejuvenate:
							colored_block_utilities_proposal_probs = colored_block_utilities_proposal_probss[
								timestep
							]
						else:
							colored_block_utilities_proposal_probs = None
					else:
						colored_block_utilities_proposal_probs = None
				(
					(temp_color_util, temp_belief, temp_env_clone, temp_agent_clone),
					temp_log_weights,
				) = update_particles(
					particles=([util], [beliefss[i]], [env_clones[i]], [agent_clones[i]]),
					log_weight=[log_weights[i, timestep - 1]],
					data=(states[timestep - 1], actions[timestep - 1], states[timestep], actions[timestep]),
					resample=resample,
					rejuvenate=rejuvenate,
					colored_block_utilities_proposal_probs=colored_block_utilities_proposal_probs,
				)
				# need for actual inference
				colored_block_utilitiess[i] = temp_color_util[0]
				beliefss[i] = temp_belief[0]
				env_clones[i] = temp_env_clone[0]
				agent_clones[i] = temp_agent_clone[0]
				log_weights[i, timestep] = temp_log_weights[0]

				# store to avoid repeated computation
				visitedParticles[(desire_int, timestep+1)] = (beliefss[i], env_clones[i], agent_clones[i], log_weights[i, timestep])

		if not output_every_timestep:
			return (colored_block_utilitiess, beliefss, env_clones, agent_clones), log_weights[:, -1]


	for timestep in tqdm.tqdm(range(1, num_timesteps)):
		if colored_block_utilities_proposal_probss is None:
			colored_block_utilities_proposal_probs = None
		else:
			if resample:
				if rejuvenate:
					colored_block_utilities_proposal_probs = colored_block_utilities_proposal_probss[
						timestep
					]
				else:
					colored_block_utilities_proposal_probs = None
			else:
				colored_block_utilities_proposal_probs = None
		(
			(colored_block_utilitiess, beliefss, env_clones, agent_clones),
			log_weights[:, timestep],
		) = update_particles(
			particles=(colored_block_utilitiess, beliefss, env_clones, agent_clones),
			log_weight=log_weights[:, timestep - 1],
			data=(states[timestep - 1], actions[timestep - 1], states[timestep], actions[timestep]),
			resample=resample,
			rejuvenate=rejuvenate,
			colored_block_utilities_proposal_probs=colored_block_utilities_proposal_probs,
		)
		if output_every_timestep:
			result.append((list(zip(colored_block_utilitiess, beliefss)), log_weights[:, timestep]))

	if output_every_timestep:
		return result
	else:
		return (colored_block_utilitiess, beliefss, env_clones, agent_clones), log_weights[:, -1]
		#return list(zip(colored_block_utilitiess, beliefss)), log_weights[:, -1]

def online_importance_sampling(
	env, states, actions, num_samples=3, colored_block_utilities_proposal_probss=None,
):
	"""Rerun importance sampling at *every* timestep and output the full sequence of posteriors.

	Args
		env (envs.construction.ConstructionEnv)
		states (list of envs.construction.State of length num_timesteps)
		actions (list of envs.construction.Action of length num_timesteps)
		num_samples (int)
		colored_block_utilities_proposal_probss (np.ndarray [num_timesteps, num_possible_rankings] or
														 [num_timesteps, num_possible_colored_blocks])

	Returns
		list of length num_timesteps where each element is the above tuple of (particles, log_weights)
			particles: list of length num_samples where each element is a tuple containing
				colored_block_utilities (dict where keys are tuples of block pairs and values are utilities)
				beliefs (list of length num_timesteps)
			log_weights (np.ndarray of shape [num_samples])
	"""

	num_timesteps = len(states)

	# Initialize weights
	log_weights = np.zeros((num_samples, num_timesteps))

	# FIRST TIMESTEP
	if colored_block_utilities_proposal_probss is None:
		colored_block_utilities_proposal_probs = None
	else:
		colored_block_utilities_proposal_probs = colored_block_utilities_proposal_probss[0:]
	(
		(colored_block_utilitiess, beliefss, env_clones, agent_clones),
		log_weights[:, 0],
	) = particle_inference(
		pickle.loads(pickle.dumps(env)),
		pickle.loads(pickle.dumps(states[:1])),
		actions[:1],
		False,
		False,
		num_samples,
		colored_block_utilities_proposal_probss=colored_block_utilities_proposal_probs,
		output_every_timestep=False,
	)
	result = [(list(zip(colored_block_utilitiess, beliefss)), log_weights[:, 0])]

	# def diagnose(res_part):
	#     for i, res in enumerate(res_part[0]):  # iterate through sampled colored blocks
	#         res_util = res[0]
	#         if res_util == env.colored_block_utilities:
	#             w = res_part[1][i]
	#             print(f"log weight is {w} probability is:")
	#             print(np.exp(w))  # print ground truth probabilitiy

	# NEXT TIMESTEPS
	for timestep in tqdm.tqdm(range(1, num_timesteps)):
		if colored_block_utilities_proposal_probss is None:
			colored_block_utilities_proposal_probs = None
		else:
			colored_block_utilities_proposal_probs = colored_block_utilities_proposal_probss[timestep:]

		(
			(colored_block_utilitiess, beliefss, env_clones, agent_clones),
			log_weights[:, timestep],
		) = particle_inference(
			pickle.loads(pickle.dumps(env)),
			pickle.loads(pickle.dumps(states[:timestep])),
			pickle.loads(pickle.dumps(actions[:timestep])),
			False,
			False,
			num_samples,
			colored_block_utilities_proposal_probss=colored_block_utilities_proposal_probs,
			output_every_timestep=False,
		)

		result.append((list(zip(colored_block_utilitiess, beliefss)), log_weights[:, timestep]))
		# diagnose(result[-1])
		# pdb.set_trace()

	return result

def get_posterior(samples, log_weights, sort_posterior=True):
	"""Convert weighted samples into a posterior

	Args:
		samples (list of length num_samples): samples[i] = (colored_block_utilities, beliefs) where
			colored_block_utilities (dict): (block1, block2) = key; utility = value
			beliefs (list of length (num_timesteps + 1)): each element = L0's belief at that time
		log_weights (list of length num_samples): list of importance / SMC weights

	Returns:
		posterior (list of length up to num_samples): the length is variable because there could
		have been duplicate elements in the samples
			posterior[i] = (colored_block_utilities_tuple, prob)
				colored_block_utilities_tuple = (colored_block_name, utility)
				prob = corresponding posterior probability
	"""
	colored_block_utilities_samples = [x[0] for x in samples]
	beliefs_samples = [x[1] for x in samples]

	# Convert to tuple
	colored_block_utilities_tuple_samples = [
		tuple(sorted(x.items())) for x in colored_block_utilities_samples
	]

	# Aggregate
	log_normalized_weights = log_weights - scipy.special.logsumexp(log_weights, axis=0)
	posterior_log_probs = {}
	for colored_block_utilities_tuple, log_normalized_weight in zip(
		colored_block_utilities_tuple_samples, log_normalized_weights
	):
		if colored_block_utilities_tuple in posterior_log_probs:
			posterior_log_probs[colored_block_utilities_tuple] = np.logaddexp(
				posterior_log_probs[colored_block_utilities_tuple], log_normalized_weight
			)
		else:
			posterior_log_probs[colored_block_utilities_tuple] = log_normalized_weight

	posterior_log_probs = {k: v for k, v in posterior_log_probs.items() if v > -np.inf}
	posterior_probs = {k: np.exp(v) for k, v in posterior_log_probs.items()}
	if sort_posterior:
		return sorted(posterior_probs.items(), key=lambda x: -x[1])
	else:
		post = list(posterior_probs.items())
		return post

def print_posterior(posterior):
	print(
		"\n".join(
			[
				f"P(util={colored_block_utilities_tuple} | states, actions) = {prob:.2f}"
				for colored_block_utilities_tuple, prob in posterior
			]
		)
	)

if __name__ == "__main__":
	import random

	seed = 5
	np.random.seed(seed)
	random.seed(seed)

	env = envs.construction_sample.sample_construction_env()
	# Sample a random layout
	L0_rollout, _ = test_sample_construction_L0_rollout.sample_L0_rollout(env, 10.0)

	# Extract a_{1:T}, s_{1:T}
	actions = [x[0] for x in L0_rollout]
	states = [x[1] for x in L0_rollout]
	state_tensor = utils.construction_data.state_to_state_tensor(states[0], 3, 'cpu')
	converted_state = utils.construction_data.state_tensor_to_state(state_tensor)
	#print(converted_state)


	# Infer p(θ, b_{1:T} | a_{1:T}, s_{1:T})
	# Importance sampling
	# samples, log_weights = importance_sample(env.clone(), states, actions, num_samples=10)
	# print_posterior(get_posterior(samples, log_weights))
	# import pdb
	#
	# pdb.set_trace()
