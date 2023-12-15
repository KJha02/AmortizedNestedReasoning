import numpy as np
import test_sample_construction_L0_rollout
import agents.construction_agent_L1
import envs.construction_sample
import pdb
import utils.construction_data
import tqdm
import utils.general
import scipy.special
import envs
import random
import _pickle as pickle
import envs.construction as construction
import agents.construction_agent_L0 as construction_agent_L0
import time

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

def init_particles(env, data, num_samples=2, other_agent_num_samples=5, seek_conflict_proposal_probs=None, beta=0.01, inference_algorithm="IS", model=None,
	prob_L1_solo=0.35):
	# Extract
	state, action = data
	log_weight = np.zeros((num_samples,))

	# Make environment and agent clones (length num_samples)
	env_clones, agent_clones = [], []

	# List of length num_samples where each element is a dict
	seek_conflictss = []

	# List of length num_samples where each element will eventually be a
	# list of length num_timesteps
	beliefss = [[] for _ in range(num_samples)]

	distinct_conflicts = [False, True]  # + construction.ALL_BLOCK_PAIRS[:env.num_possible_block_pairs]
	full_action_probss = []
	#print(f"Action: {action}")
	# assert num_samples == 2

	visitedParticles = {}
	belief = None

	for sample_id in range(num_samples):
		if seek_conflict_proposal_probs is not None and num_samples < 2:
			seek_conflict = np.random.choice(np.array(distinct_conflicts), p=seek_conflict_proposal_probs)
		else:
			seek_conflict = distinct_conflicts[sample_id]
		# seek_conflict = distinct_conflicts[sample_id]
		seek_conflictss.append(seek_conflict)

		env_clone = pickle.loads(pickle.dumps(env))

		if seek_conflict == True or seek_conflict == False:
			env_clone.base_colored_block_utilities_L1 = None  # temporary measure so that we just do helping hurting
			env_clone.seek_conflict = seek_conflict
		else:  # convert tuple goal into 
			artificial_util_prior = [0.0] * env_clone.num_possible_block_pairs
			artificial_util_prior[construction.ALL_BLOCK_PAIRS.index(seek_conflict)] = 1
			L1_sampled_util = envs.construction_sample.sample_block_pair_utilities(env_clone.num_possible_block_pairs, prior=artificial_util_prior)
			env_clone.base_colored_block_utilities_L1 = L1_sampled_util
			env_clone.seek_conflict = random.choice([False, True])

		env_clones.append(env_clone)

		if model is not None:
			model.eval()
		agent_clones.append(
			agents.construction_agent_L1.AgentL1(
				seek_conflict=seek_conflict,
				base_colored_block_utilities=env_clone.base_colored_block_utilities_L1,
				num_possible_block_pairs=env_clone.num_possible_block_pairs,
				initial_state_L0=env_clone.initial_state_L0,
				initial_agent_location_L1=env_clone.initial_agent_location_L1,
				initial_block_picked=env_clone.initial_agent_inv_L1,
				ground_truth_colored_block_utilities_L0=env_clone.colored_block_utilities_L0,
				beta_L0=env_clone.beta_L0,
				beta_L1=env_clone.beta_L1,
				transition_L1=env_clone.transition,
				transition_L0=env_clone.env_L0.transition,
				inference_algorithm=inference_algorithm,
				num_samples=other_agent_num_samples,
				model=model,
				visitedParticles=visitedParticles
			)
		)
		# this is the prior
		if seek_conflict_proposal_probs is not None:
			seek_conflict_proposal_prob = seek_conflict_proposal_probs[sample_id]
		# else:
		# 	seek_conflict_proposal_prob = 1 / num_samples

		# PROPOSE
		# Sample o_1 ~ p(o | s_1)
		# pdb.set_trace()
		obs = construction.ObservationL1(state.observation_L0, construction.Action.STOP)
		agent_clones[sample_id].observations.append(obs)
		# pdb.set_trace()
		# Sample b_1 ~ p(b | ...)
		if sample_id == 0:
			try:
				belief = agent_clones[sample_id].get_belief()
			except:
				pdb.set_trace()
		else:
			agent_clones[sample_id].particles = agent_clones[0].particles
		# for (util, prob) in belief:
		# 	max_pair = max(dict(util), key=dict(util).get)
		# 	print(f"Sample ID: {sample_id}; Utility: {max_pair}; Probability: {prob}")


		# try:
		#     belief = agent_clones[sample_id].get_belief()
		# except:
		#     pdb.set_trace()
		beliefss[sample_id].append(belief) 

		# WEIGH based on p(a_1 | ...)p(s | ...)
		# Score p(a_1 | ...)
		# pdb.set_trace()
		try:

			full_action_probs = agent_clones[sample_id].get_action_probs(agent_location_L1=agent_clones[sample_id].curr_state_L1.agent_location_L1,
															   belief=belief)
		except:
			pdb.set_trace()
			full_action_probs = agent_clones[sample_id].get_action_probs(agent_location_L1=agent_clones[sample_id].curr_state_L1.agent_location_L1,
															   belief=belief)
		full_action_probss.append(full_action_probs)
		action_prob = full_action_probs[action.value]
		if seek_conflict_proposal_probs is None or True:
			if action_prob < 1e-6:
				log_weight[sample_id] = -1e6
			else:
				log_weight[sample_id] = np.log(action_prob)
		else:
			seek_conflict_proposal_prob = max(1e-6, seek_conflict_proposal_prob)
			action_prob = max(1e-6, action_prob)
			# log_weight[sample_id] = np.log(seek_conflict_proposal_prob)
			log_weight[sample_id] = (
				np.log(action_prob)
				+ np.log(len(seek_conflict_proposal_probs))
				- np.log(seek_conflict_proposal_prob)
			)
		visitedParticles = agent_clones[-1].prev_sampled_utilities

	# for agent in agent_clones:
	# 	agent.prev_sampled_utilities = visitedParticles

	log_weight, too_low = get_corrected_log_weight(log_weight)

	particles = seek_conflictss, beliefss, env_clones, agent_clones
	return particles, log_weight, full_action_probss


def update_particles(
	particles,
	log_weight,
	data,
	resample=True,
	rejuvenate=False,
	seek_conflict_proposal_probs=None,
):
	"""Go from (θ, b_{1:t - 1}) to (θ, b_{1:t}) with or without resampling.
	Used for importance sampling and SMC.

	Args
		particles
			seek_conflictss: List of length num_samples where each element is a bool
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
		seek_conflict_proposal_probs (None or np.array or list of floats)

	Returns
		updated_particles
			updated_seek_conflictss: List of length num_samples where each element is a bool
			updated_beliefss: List of length num_samples where each element is list of length
				(num_timesteps + 1)
			updated_env_clones: list of length num_samples
			updated_agent_clones: list of length num_samples
		updated_log_weight [num_samples]
	"""
		# Extract
	num_samples = len(log_weight)
	seek_conflictss, beliefss, env_clones, agent_clones = particles
	prev_state, prev_action, state, action = data
	# pdb.set_trace()
	updated_log_weight = np.zeros((num_samples,))
	num_timesteps = len(beliefss[0])
	full_action_probss = []
	for sample_id in range(num_samples):
		assert len(beliefss[sample_id]) == num_timesteps
	if resample:
		# RESAMPLE
		# Sample ancestral indices
		ancestral_index = utils.general.sample_ancestral_index(log_weight)

		# Reassign utilities
		seek_conflictss_new = []
		for sample_id in range(num_samples):
			seek_conflictss_new.append(seek_conflictss[ancestral_index[sample_id]])
		seek_conflictss = seek_conflictss_new

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
	if rejuvenate and  seek_conflict_proposal_probs is not None:
		# REJUVENATE θ based on q(θ | s_{1:t}, a_{1:t})
		# At the moment, we sample directly from q without any correction.
		for sample_id in range(num_samples):
			import envs.construction_sample
			seek_conflict = envs.construction_sample.sample_seek_conflict_value(
				prior=seek_conflict_proposal_probs,
				return_prob=False
			)
			seek_conflictss[sample_id] = seek_conflict

			env_clones[sample_id].seek_conflict = seek_conflictss[sample_id]
			agent_clones[sample_id].seek_conflict = seek_conflictss[sample_id]

	#print("Updating L2 particles")
	visitedParticles = agent_clones[-1].prev_sampled_utilities
	belief = None
	for sample_id in range(num_samples):
		# TODO: Step the environment so that timestep updates
		env_clones[sample_id].agent_location_L1 = prev_state.agent_location_L1
		env_clones[sample_id].agent_inv_L1 = prev_state.agent_inv_L1

		#print(f"Sample {sample_id} timestep = {env_clones[sample_id].timestep}")
		# PROPOSE
		# Sample o_t ~ p(o | ...)
		# pdb.set_trace()
		obs = construction.ObservationL1(state_L0=prev_state.observation_L0, action_L0=prev_state.prev_action_L0)
		agent_clones[sample_id].observations.append(obs)
		agent_clones[sample_id].curr_state_L1 = prev_state  # L1's state at timestep t-1
		agent_clones[sample_id].curr_state_L0 = prev_state.observation_L0  # L2's state at timestep t-1
		agent_clones[sample_id].prev_sampled_utilities = visitedParticles
		try:
			assert len(beliefss[sample_id]) == num_timesteps
		except:
			pdb.set_trace()
				# Sample b_t ~ p(b | ...)

		# belief = agent_clones[sample_id].get_belief()
		if sample_id == 0:
			try:
				belief = agent_clones[sample_id].get_belief()
			except:
				pdb.set_trace()
		else:
			agent_clones[sample_id].particles = agent_clones[0].particles

		# if agent_clones[sample_id].num_samples == 5 and num_timesteps > 1:
		# 	b_utils = [dict(x[0]) for x in belief]
		# 	gt = ('#', '%')
		# 	if gt not in b_utils[0]:
		# 		gt = ('%', '#')

		# 	for gt_idx, b_u in enumerate(b_utils):
		# 		if b_u[gt] == 100:
		# 			belief_probs = [x[1] for x in belief]
		# 			pdb.set_trace()
		# 			break
		# try:
		# 	#print("Probabilities for L1 get belief (in L2 inference)")
		# 	belief = agent_clones[sample_id].get_belief()
		# 	#print("-------------\n\n")
		# except:
		# 	pdb.set_trace()
		# 	belief = agent_clones[sample_id].get_belief()
		# for (util, prob) in belief:
		# 	max_pair = max(dict(util), key=dict(util).get)
		# 	print(f"Sample ID: {sample_id}; Utility: {max_pair}; Probability: {prob}")

		beliefss[sample_id].append(belief)

		# WEIGH based on p(a_t | ...)p(s_t | ...)
		# Score p(a_t | ...)
		# pdb.set_trace()
		full_action_probs = agent_clones[sample_id].get_action_probs(agent_location_L1=agent_clones[sample_id].curr_state_L1.agent_location_L1,
															   belief=beliefss[sample_id][-1])
		full_action_probss.append(full_action_probs)
		action_prob = full_action_probs[action.value]
		# env_clones[sample_id].timestep += 1

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
			# TODO: just try without adding the uniform metric
			# updated_log_weight[sample_id] = np.log(np.exp(log_weight[sample_id]) + np.exp(log_action_prob))
			# TODO: just try downweighting first product so that it moves to average of first and second product
			# first_weight = max(0.5, alpha * 0.99 ** num_timesteps)
			# updated_log_weight[sample_id] = first_weight*np.log(alpha * np.exp(log_weight[sample_id]) + (1 - alpha) * p_uniform) + (1-first_weight)*log_action_prob
			# TODO: adjust alpha
			# TODO: normalize over time (average first and second)
			first_prod = np.log(alpha * np.exp(log_weight[sample_id]) + (1 - alpha) * p_uniform)
			# updated_log_weight[sample_id] = (first_prod + log_action_prob) / 2
			# TODO: normalize without uniform term
			# updated_log_weight[sample_id] = (log_weight[sample_id] + log_action_prob) / 2
			# TODO: Standard version
			updated_log_weight[sample_id] = first_prod + log_action_prob

		# Assert
		for i in range(sample_id):
			try:
				assert len(beliefss[i]) == num_timesteps + 1
			except:
				pdb.set_trace()
		for i in range(sample_id + 1, num_samples):
			try:
				assert len(beliefss[i]) == num_timesteps
			except:
				pdb.set_trace()
		env_clones[sample_id].timestep += 1
		visitedParticles = agent_clones[sample_id].prev_sampled_utilities
	# Assert
	
	# for si in range(len(agent_clones[0].observations)):
	# 	try:
	# 		assert agent_clones[0].states_L0[si] == agent_clones[1].states_L0[si]
	# 		assert agent_clones[0].actions_L0[si] == agent_clones[1].actions_L0[si]
	# 	# assert agent_clones[0].observations == agent_clones[1].observations
	# 	except:
	# 		pdb.set_trace()

	# for sample_id in range(num_samples):
	# 	assert len(beliefss[sample_id]) == num_timesteps + 1

	updated_particles = seek_conflictss, beliefss, env_clones, agent_clones
	# Weight correction
	corrected_log_weight, too_low = get_corrected_log_weight(updated_log_weight)
	return updated_particles, corrected_log_weight, full_action_probss

def L1_particle_inference(
	env,
	states,
	actions,
	resample=False,
	rejuvenate=False,
	num_samples_L2=2,
	num_samples_L1=5,
	seek_conflict_proposal_probss=None,
	other_agent_inference_algorithm="IS",
	output_every_timestep=False,
	other_agent_inference_model=None,
	visitedParticles=None,
	beta_L0=None,
	beta_L1=None
):
	if beta_L0 is not None:
		env.beta_L0=beta_L0
	if beta_L1 is not None:
		env.beta_L1 = beta_L1

	num_timesteps = len(states)
	log_weights = np.zeros((num_samples_L2, num_timesteps))
	if seek_conflict_proposal_probss is None:
		seek_conflict_proposal_probs = None
	else:
		seek_conflict_proposal_probs = seek_conflict_proposal_probss[0]
	(seek_conflictss, beliefss, env_clones, agent_clones), log_weights[:, 0], L1_action_probss = init_particles(
		env,
		data=(states[0], actions[0]),
		num_samples=num_samples_L2,
		other_agent_num_samples=num_samples_L1,
		seek_conflict_proposal_probs=seek_conflict_proposal_probs,
		inference_algorithm=other_agent_inference_algorithm,
		model=other_agent_inference_model
	)

	if output_every_timestep:
		result = [((seek_conflictss, beliefss, env_clones, agent_clones), log_weights[:, 0])]


	if visitedParticles is not None:
		for i, conflict in enumerate(seek_conflictss):
			if conflict in visitedParticles:
				belief, env_clone, agent_clone, log_weight, start_time = visitedParticles[conflict]
				beliefss[i] = belief
				env_clones[i] = env_clone
				agent_clones[i] = agent_clone
				log_weights[i, start_time - 1] = log_weight
			else:
				start_time = 1

			for timestep in tqdm.tqdm(range(start_time, num_timesteps)):  # only update a single particle from where it left off
				if seek_conflict_proposal_probss is None:
					seek_conflict_proposal_probs = None
				else:
					if resample:
						if rejuvenate:
							seek_conflict_proposal_probs = seek_conflict_proposal_probss[timestep]
						else:
							seek_conflict_proposal_probs = None
					else:
						seek_conflict_proposal_probs = None
				(
					(temp_conflicts, temp_belief, temp_env_clone, temp_agent_clone),
					temp_log_weights,
					temp_L1_action_probss
				) = update_particles(
					particles=([conflict], [beliefss[i]], [env_clones[i]], [agent_clones[i]]),
					log_weight=[log_weights[i, timestep - 1]],
					data=(states[timestep], actions[timestep - 1], states[timestep], actions[timestep]),
					resample=resample,
					rejuvenate=rejuvenate,
					seek_conflict_proposal_probs=seek_conflict_proposal_probs,
				)
				seek_conflictss[i] = temp_conflicts[0]
				beliefss[i] = temp_belief[0]
				env_clones[i] = temp_env_clone[0]
				agent_clones[i] = temp_agent_clone[0]
				log_weights[i, timestep] = temp_log_weights[0]
		if not output_every_timestep:
			return (seek_conflictss, beliefss, env_clones, agent_clones), log_weights[:, -1]



	for timestep in tqdm.tqdm(range(1, num_timesteps)):
	# for timestep in tqdm.tqdm(range(1, 4)):
		if seek_conflict_proposal_probss is None:
			seek_conflict_proposal_probs = None
		else:
			if resample:
				if rejuvenate:
					seek_conflict_proposal_probs = seek_conflict_proposal_probss[timestep]
				else:
					seek_conflict_proposal_probs = None
			else:
				seek_conflict_proposal_probs = None
		(
			(seek_conflictss, beliefss, env_clones, agent_clones),
			log_weights[:, timestep],
			L1_action_probss,
		) = update_particles(
			particles=(seek_conflictss, beliefss, env_clones, agent_clones),
			log_weight=log_weights[:, timestep - 1],
			data=(states[timestep], actions[timestep - 1], states[timestep], actions[timestep]),
			resample=resample,
			rejuvenate=rejuvenate,
			seek_conflict_proposal_probs=seek_conflict_proposal_probs,
		)
		if output_every_timestep:
			result.append(((seek_conflictss, beliefss, env_clones, agent_clones), log_weights[:, timestep]))

	if output_every_timestep:
		return result
	else:
		return (seek_conflictss, beliefss, env_clones, agent_clones), log_weights[:, -1]


def L1_online_importance_sampling(
	env, states, actions, num_samples_L2=2, num_samples_L1=5, seek_conflict_proposal_probss=None, 
	other_agent_inference_algorithm="IS", other_agent_inference_model=None, timeRollout=False
):
	'''
	Env should be construction L1 env
	States should be L1 states that include L0 prev actions
	Actions should be L1 action history
	'''

	num_timesteps = len(states)
	rolloutRuntime = []

	# Initialize weights
	log_weights = np.zeros((2, num_timesteps))
	# FIRST TIMESTEP
	if seek_conflict_proposal_probss is None:
		seek_conflict_proposal_probs = None
	else:
		seek_conflict_proposal_probs = seek_conflict_proposal_probss[0:]


	if timeRollout:
		start = time.time()


	(
		(seek_conflictss, beliefss, env_clones, agent_clones),
		log_weights[:, 0],
	) = L1_particle_inference(
		pickle.loads(pickle.dumps(env)),
		pickle.loads(pickle.dumps(states[:1])),
		pickle.loads(pickle.dumps(actions[:1])),
		resample=False,
		rejuvenate=False,
		num_samples_L2=num_samples_L2,
		num_samples_L1=num_samples_L1,
		seek_conflict_proposal_probss=seek_conflict_proposal_probs,
		other_agent_inference_algorithm=other_agent_inference_algorithm,
		other_agent_inference_model=other_agent_inference_model,
		output_every_timestep=False,
	)

	if timeRollout:
		rolloutRuntime.append(time.time() - start)

	result = [((seek_conflictss, beliefss, env_clones, agent_clones), log_weights[:, 0])]


	visitedParticles = {}
	for i, conflict in enumerate(seek_conflictss):
		infoToStore = (beliefss[i], env_clones[i], agent_clones[i], log_weights[int(conflict), 0], 1)
		visitedParticles[conflict] = infoToStore


	# NEXT TIMESTEPS

	for timestep in tqdm.tqdm(range(1, num_timesteps)):
		if seek_conflict_proposal_probss is None:
			seek_conflict_proposal_probs = None
		else:
			seek_conflict_proposal_probs = seek_conflict_proposal_probss[timestep:]

		if timeRollout:
			start = time.time()

		(
			(seek_conflictss, beliefss, env_clones, agent_clones),
			log_weights[:, timestep],
		) = L1_particle_inference(
			pickle.loads(pickle.dumps(env)),
			pickle.loads(pickle.dumps(states[:timestep])),
			pickle.loads(pickle.dumps(actions[:timestep])),
			resample=False,
			rejuvenate=False,
			num_samples_L2=num_samples_L2,
			num_samples_L1=num_samples_L1,
			seek_conflict_proposal_probss=seek_conflict_proposal_probs,
			other_agent_inference_algorithm=other_agent_inference_algorithm,
			other_agent_inference_model=other_agent_inference_model,
			output_every_timestep=False,
			visitedParticles=visitedParticles
		)

		if timeRollout:
			rolloutRuntime.append(time.time() - start)

		for i, conflict in enumerate(seek_conflictss):
			infoToStore = (beliefss[i], env_clones[i], agent_clones[i], log_weights[int(conflict), 0], timestep+1)
			visitedParticles[conflict] = infoToStore
		result.append(((seek_conflictss, beliefss, env_clones, agent_clones), log_weights[:, timestep]))
	return result, rolloutRuntime


def get_gt_L1_is_inference(rollout_env, states_raw, actions_raw, num_samples_L2=2, num_samples_L1=45, beta_L0=None, beta_L1 = None):
	all_inferences = L1_particle_inference(rollout_env, states_raw, actions_raw, num_samples_L2=num_samples_L2, num_samples_L1=num_samples_L1, output_every_timestep=True,
		beta_L0=beta_L0, beta_L1=beta_L1)
	IS_inferences = L1_inference_to_posterior_distrib(all_inferences)
	# for j in range(len(all_inferences)):
	# 	seek_conflictss, beliefss, env_clones, agent_clones = all_inferences[j][0]
	# 	posterior_belief = get_posterior(
	# 		list(zip(seek_conflictss, beliefss)), all_inferences[j][1]
	# 	)
	# 	posterior_distrib = [None] * len(posterior_belief)
	# 	for p in posterior_belief:  # doing this to preserve order in predictions
	# 		inferred_conflict_idx = int(p[0])  # going from conflict to index for consistency
	# 		inferred_conflict_pred = p[1]  # what is the actual probability assigned to this belief
	# 		posterior_distrib[inferred_conflict_idx] = inferred_conflict_pred
	# 	IS_inferences.append(posterior_distrib)

	return IS_inferences

def L1_inference_to_posterior_distrib(inference):
	result = []
	for j in range(len(inference)):
		seek_conflictss, beliefss, env_clones, agent_clones = inference[j][0]
		posterior_belief = get_posterior(
			list(zip(seek_conflictss, beliefss)), inference[j][1]
		)
		posterior_distrib = [None] * 2
		for p in posterior_belief:  # doing this to preserve order in predictions
			inferred_conflict_idx = int(p[0])  # going from conflict to index for consistency
			inferred_conflict_pred = p[1]  # what is the actual probability assigned to this belief
			posterior_distrib[inferred_conflict_idx] = inferred_conflict_pred

		if posterior_distrib[0] is None or posterior_distrib[0] == 0:
			posterior_distrib[0] = 1.0 - posterior_distrib[1]
		if posterior_distrib[1] is None or posterior_distrib[1] == 0:
			posterior_distrib[1] = 1.0 - posterior_distrib[0]


		result.append(posterior_distrib)

	return result


def get_posterior(samples, log_weights, sort_posterior=True):
	"""Convert weighted samples into a posterior

	Args:
		samples (list of length num_samples): samples[i] = (seek_conflict, beliefs) where
			seek_conflict = True or False
			beliefs (list of length (num_timesteps + 1)): each element = L0's belief at that time
		log_weights (list of length num_samples): list of importance / SMC weights

	Returns:
		posterior (list of length up to num_samples): the length is variable because there could
		have been duplicate elements in the samples
			posterior[i] = (seek_conflict, prob)
				seek_conflict = (colored_block_name, utility)
				prob = corresponding posterior probability
	"""
	seek_conflict_samples = [x[0] for x in samples]
	belief_samples = [x[1] for x in samples]
	# Aggregate
	log_normalized_weights = log_weights - scipy.special.logsumexp(log_weights, axis=0)
	posterior_log_probs = {}
	for seek_conflict, log_normalized_weight in zip(
		seek_conflict_samples, log_normalized_weights
	):
		if seek_conflict in posterior_log_probs:
			posterior_log_probs[seek_conflict] = np.logaddexp(
				posterior_log_probs[seek_conflict], log_normalized_weight
			)
		else:
			posterior_log_probs[seek_conflict] = log_normalized_weight

	posterior_log_probs = {k: v for k, v in posterior_log_probs.items() if v > -np.inf}
	posterior_probs = {k: np.exp(v) for k, v in posterior_log_probs.items()}
	if sort_posterior:
		return sorted(posterior_probs.items(), key=lambda x: -x[1])
	else:
		post = list(posterior_probs.items())
		return post
