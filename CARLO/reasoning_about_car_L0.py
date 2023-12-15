from scenario import Scenario1, Action, get_partial_states
from drivingAgents.car_agent_L0 import AgentL0, locToGoalString
import random
import numpy as np
import _pickle as pickle
import pdb
import time
import tqdm
import scipy.special
from car_utils.car_sampler import sample_lane_utilities

GOAL_SPACE = ['forward', 'left', 'right']

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


def init_particles(env, data, num_samples=3, lane_utilities_proposal_probs=None, carExistPrior=0.5, beta=0.01, 
	sampled_actions=10, lookAheadDepth=5, history=None, 
	full_action_dict=None, state_belief_model=None, exist_belief_model=None, inference_algorithm="Online_IS+NN"):
	'''
	Assuming the state extracted from the data is the fully observable Scenario.w World object
	'''
	action_space = list(Action)
	state, action, agentID = data
	action_idx = action_space.index(Action(action.value))
	log_weight = np.zeros((num_samples,))
	# Make agent clone (length num_samples)
	agent_clones = []
	# List of length num_samples where each element is a target lane value (0 through 15 inclusive)
	lane_utilitiess = []
	# List of length num_samples where each element will eventually be a
	# list of length num_timesteps
	beliefss = [[] for _ in range(num_samples)]

	distinct_lanes= ["forward", "left", "right"]
	num_possible_lane_targets = len(distinct_lanes)


	next_action_probs = {}


	topKLanes = []
	if lane_utilities_proposal_probs is not None:
		sortedLaneProbs = sorted(lane_utilities_proposal_probs, reverse=True)
		lane_utilities_proposal_probs = list(lane_utilities_proposal_probs)

		for idx in range(num_samples):
			trueDesireInt = lane_utilities_proposal_probs.index(sortedLaneProbs[idx])
			lane_utilities = distinct_lanes[trueDesireInt]
			probability = sortedLaneProbs[idx]
			topKLanes.append((lane_utilities, probability))

	for sample_id in range(num_samples):
		# Sample θ ~ p(θ)
		if lane_utilities_proposal_probs is None:
			if sample_id < len(distinct_lanes) and num_samples >= num_possible_lane_targets and inference_algorithm != 'random':
				lane_utilities = distinct_lanes[sample_id] 
			else:  # if we have more samples than distinct utilities or we aren't fully sampling all particles
				lane_utilities = np.random.choice(distinct_lanes)
		else:
			if sample_id < len(topKLanes):
				(lane_utilities, lane_utilities_proposal_prob) = topKLanes[sample_id]
			else:
				(
					lane_utilities,
					lane_utilities_proposal_prob,
				) = sample_lane_utilities(
					prior=lane_utilities_proposal_probs,
					return_prob=True,
				)

		lane_utilitiess.append(lane_utilities)

		try:
			if (lane_utilities, 1) in history:
				beliefss.append(None)
				env_clones.append(None)
				agent_clones.append(None)
				continue
		except:
			pass 

		clonedAgent = AgentL0(
			agentID=agentID,
			goalActionString=lane_utilities,
			transition=env.transition,
			initialLocMapping=pickle.loads(pickle.dumps(env.initialLocMapping)),
			carExistPrior=carExistPrior,
			beta=beta,
			sampled_actions=sampled_actions, 
			lookAheadDepth=lookAheadDepth,
			state_model=state_belief_model,
			exist_model=exist_belief_model,
			inference_algorithm=inference_algorithm
		)
		agent_clones.append(clonedAgent)

		#env.reset()
		# pdb.set_trace()
		partialObs = get_partial_states(state, id=agentID)  # get the partial observation just for this agent
		agent_clones[sample_id].observations.append(partialObs)
		if full_action_dict is not None:  # we do this so that we can use the neural net predictions
			agent_clones[sample_id].full_action_history.append(full_action_dict)

		try:
			belief, belief_tensor = agent_clones[sample_id].get_belief()
		except:
			pdb.set_trace()
		beliefss[sample_id].append(belief)

		# WEIGH based on p(a_1 | ...)p(s | ...)
		# Score p(a_1 | ...)
		action_prob = agent_clones[sample_id].get_action_probs(belief=belief)[action_idx]

		# Assign the weight
		# if lane_utilities_proposal_probs is None or True:  # ignore the else statement for now
		if action_prob < 1e-6:
			log_weight[sample_id] = -1e6
		else:
			log_weight[sample_id] = np.log(action_prob)


		# temporary stepping ahead 
		temp_action_dict = {}
		for a in partialObs.dynamic_agents: # assume other agents are going to maintain their previous actions
			temp_action_dict[a.ID] = a.prev_action
		temp_action_dict[agentID] = action
		next_partial_obs = agent_clones[sample_id].transition(partialObs, temp_action_dict)

		# build belief so you can do inference over next action probabilities
		agent_clones[sample_id].observations.append(next_partial_obs)
		agent_clones[sample_id].full_action_history.append(temp_action_dict)
		next_belief, next_belief_tensor = agent_clones[sample_id].get_belief()
		next_action_prob = agent_clones[sample_id].get_action_probs(belief=next_belief)
		next_action_probs[lane_utilities] = next_action_prob  # store 

		next_partial_obs = agent_clones[sample_id].observations.pop()
		agent_clones[sample_id].full_action_history.pop()
		del next_partial_obs
		del temp_action_dict

		


	# Weight correction
	log_weight, too_low = get_corrected_log_weight(log_weight)

	particles = lane_utilitiess, beliefss, agent_clones
	return particles, log_weight, next_action_probs

def update_particles(particles, log_weight, data, lane_utilities_proposal_probs=None, full_action_dict=None):
	'''
	Assuming the state extracted from the data is the fully observable Scenario.w World object
	'''
	action_space = list(Action)
	try:
		num_samples = len(log_weight)
	except:
		pdb.set_trace()
	lane_utilitiess, beliefss, agent_clones = particles
	prev_state, prev_action, state, action, agentID = data
	action_idx = action_space.index(Action(action.value))
	updated_log_weight = np.zeros((num_samples,))
	num_timesteps = len(beliefss[0])


	next_action_probs = {}

	for sample_id in range(num_samples):
		partialObs = get_partial_states(state, id=agentID)  # get the partial observation just for this agent
		notVisible = False
		if type(partialObs) == list:  # temporarily append previous belief
			partialObs = agent_clones[sample_id].observations[-1]
			notVisible = True
		if not notVisible:  # only save observations we actually see or those artifically sampled ahead
			agent_clones[sample_id].observations.append(partialObs)
			if full_action_dict is not None:  # we do this so that we can use the neural net predictions
				agent_clones[sample_id].full_action_history.append(full_action_dict)
		if notVisible:  # if we can't see the agent, keep previous belief
			belief = beliefss[sample_id][-1]
		else:  
			belief, belief_tensor = agent_clones[sample_id].get_belief()

		beliefss[sample_id].append(belief)

		# WEIGH based on p(a_1 | ...)p(s | ...)
		# Score p(a_1 | ...)
		if notVisible:  # if we can't see the agent copy the goal inference from previous time
			updated_log_weight[sample_id] = log_weight[sample_id]  
		else: # if we are able to see the agent, update belief
			action_prob = agent_clones[sample_id].get_action_probs(belief=belief)[action_idx]

			# Assign the weight
			if action_prob < 1e-6:
				log_action_prob = -1e6
			else:
				log_action_prob = np.log(action_prob)

			alpha = 0.999
			p_uniform = 1 / num_samples
			updated_log_weight[sample_id] = log_weight[sample_id] + log_action_prob
			#updated_log_weight[sample_id] = np.log(alpha * np.exp(log_weight[sample_id]) + (1 - alpha) * p_uniform) + log_action_prob

		# temporary stepping ahead 
		temp_action_dict = {}
		for a in partialObs.dynamic_agents: # assume other agents are going to maintain their previous actions
			temp_action_dict[a.ID] = a.prev_action
		temp_action_dict[agentID] = action
		next_partial_obs = agent_clones[sample_id].transition(partialObs, temp_action_dict)

		# build belief so you can do inference over next action probabilities
		agent_clones[sample_id].observations.append(next_partial_obs)
		agent_clones[sample_id].full_action_history.append(temp_action_dict)
		next_belief, next_belief_tensor = agent_clones[sample_id].get_belief()
		next_action_prob = agent_clones[sample_id].get_action_probs(belief=next_belief)
		next_action_probs[lane_utilitiess[sample_id]] = next_action_prob  # store 
		if not notVisible:  # we want to artificially step ahead cars we can't see
			next_partial_obs = agent_clones[sample_id].observations.pop()
			agent_clones[sample_id].full_action_history.pop()
		del next_partial_obs
		del temp_action_dict

	updated_particles = lane_utilitiess, beliefss, agent_clones
	# Weight correction
	corrected_log_weight, too_low = get_corrected_log_weight(updated_log_weight)

	return updated_particles, corrected_log_weight, next_action_probs



def particle_inference(
	agentID,
	env,
	states, 
	actions, 
	num_samples=3, 
	lane_utilities_proposal_probss=None, 
	output_every_timestep=False,
	visitedParticles=None,
	carExistPrior=0.5,
	sampled_actions=10, 
	lookAheadDepth=5,
	beta=0.01,
	full_action_history=[], 
	state_belief_model=None, 
	exist_belief_model=None,
	inference_algorithm="Online_IS+NN"
):
	'''
	agentID is the target agent we want to do inference over
	env is scenario environment
	States is a list of fully observable Scenario.w World objects
	actions is the action history for agent agentID
	lane_utilities_proposal_probss is a list of neural net proposals for every timestep
	'''
	num_timesteps = len(states)
	num_possible_lane_targets = 3
	num_possible_actions = 5
	# Initialize weights
	log_weights = np.zeros((num_samples, num_timesteps))
	next_action_weights = {}



	if lane_utilities_proposal_probss is None:
		lane_utilities_proposal_probs = None 
	else:
		lane_utilities_proposal_probs = lane_utilities_proposal_probss[0]
	if len(full_action_history) == 0:
		action_dict = None
	else:
		action_dict = full_action_history[0]

	(lane_utilitiess, beliefss, agent_clones), log_weights[:, 0], next_action_probs = init_particles(
		env,
		data=(states[0], actions[0], agentID),
		num_samples=num_samples,
		lane_utilities_proposal_probs=lane_utilities_proposal_probs,
		history=visitedParticles,
		carExistPrior=carExistPrior,
		sampled_actions=sampled_actions, 
		lookAheadDepth=lookAheadDepth,
		beta=beta,
		full_action_dict=action_dict,
		state_belief_model=state_belief_model,
		exist_belief_model=exist_belief_model,
		inference_algorithm=inference_algorithm
	)

	next_action_weights = {0: next_action_probs}



	if output_every_timestep:
		result = [((lane_utilitiess, beliefss, agent_clones), log_weights[:, 0])]

	if visitedParticles is not None:  # key is (integer of goal, start_time)
		for i, util in enumerate(lane_utilitiess):
			desire_goal = util

			start_time = None
			for t in range(num_timesteps, 0, -1):
				if (desire_goal, t) in visitedParticles:  # load in existing particle if it was sampled
					start_time = t
					belief, agent_clone, log_weight, next_action_prob = visitedParticles[(desire_goal,t)]
					beliefss[i] = belief
					agent_clones[i] = agent_clone
					log_weights[i, start_time - 1] = log_weight
					break
			if start_time is None:  # otherwise we update from beginning
				start_time = 1

			for timestep in tqdm.tqdm(range(start_time, num_timesteps)):  # only update a single particle from where it left off
				lane_utilities_proposal_probs = None
				if len(full_action_history) == 0:
					action_dict = None
				else:
					action_dict = full_action_history[timestep]
				(
					(temp_lane_util, temp_belief, temp_agent_clone),
					temp_log_weights, temp_next_action_probs
				) = update_particles(
					particles=([util], [beliefss[i]], [agent_clones[i]]),
					log_weight=[log_weights[i, timestep - 1]],
					data=(states[timestep - 1], actions[timestep - 1], states[timestep], actions[timestep], agentID),
					lane_utilities_proposal_probs=lane_utilities_proposal_probs,
					full_action_dict=action_dict
				)
				# need for actual inference
				# we only index into 0 because it return an array in the form [[x]], so we remove the extra dimension
				lane_utilitiess[i] = temp_lane_util[0]
				beliefss[i] = temp_belief[0]
				agent_clones[i] = temp_agent_clone[0]
				log_weights[i, timestep] = temp_log_weights[0]
				if timestep not in next_action_weights:
					next_action_weights[timestep] = {}
				try:
					next_action_weights[timestep][lane_utilitiess[i]] = temp_next_action_probs[lane_utilitiess[i]]
				except:
					pdb.set_trace()

				# store to avoid repeated computation
				visitedParticles[(desire_goal, timestep+1)] = (beliefss[i], agent_clones[i], log_weights[i, timestep], next_action_weights[timestep][lane_utilitiess[i]])


		if not output_every_timestep:
			return (lane_utilitiess, beliefss, agent_clones), log_weights[:, -1], next_action_weights

	for timestep in tqdm.tqdm(range(1, num_timesteps)):
		lane_utilities_proposal_probs = None
		if len(full_action_history) == 0:
			action_dict = None
		else:
			action_dict = full_action_history[timestep]
		(
			(lane_utilitiess, beliefss, agent_clones),
			log_weights[:, timestep], next_action_probs
		) = update_particles(
			particles=(lane_utilitiess, beliefss, agent_clones),
			log_weight=log_weights[:, timestep - 1],
			data=(states[timestep - 1], actions[timestep - 1], states[timestep], actions[timestep], agentID),
			lane_utilities_proposal_probs=lane_utilities_proposal_probs,
			full_action_dict=action_dict
		)
		next_action_weights[timestep] = next_action_probs
		if output_every_timestep:
			result.append(((lane_utilitiess, beliefss, agent_clones), log_weights[:, timestep]))


	if output_every_timestep:
		return result, next_action_weights
	else:
		return (lane_utilitiess, beliefss, agent_clones), log_weights[:, -1], next_action_weights



def get_posterior(samples, log_weights, sort_posterior=True):
	lane_utilities_samples = [x[0] for x in samples]
	beliefs_samples = [x[1] for x in samples]

	# Aggregate
	log_normalized_weights = log_weights - scipy.special.logsumexp(log_weights, axis=0)
	posterior_log_probs = {}
	for lane_utility, log_normalized_weight in zip(
		lane_utilities_samples, log_normalized_weights
	):
		if lane_utility in posterior_log_probs:
			posterior_log_probs[lane_utility] = np.logaddexp(
				posterior_log_probs[lane_utility], log_normalized_weight
			)
		else:
			posterior_log_probs[lane_utility] = log_normalized_weight

	posterior_log_probs = {k: v for k, v in posterior_log_probs.items() if v > -np.inf}
	posterior_probs = {k: np.exp(v) for k, v in posterior_log_probs.items()}
	if sort_posterior:
		return sorted(posterior_probs.items(), key=lambda x: -x[1])
	else:
		post = list(posterior_probs.items())
		return post

def car_goal_inference_to_posterior_distrib(inference):
	result = []
	possible_goals = ["forward", "left", "right"]
	for j in range(len(inference)):
		lane_utilitiess, beliefss, agent_clones = inference[j][0]
		posterior_belief = get_posterior(
			list(zip(lane_utilitiess, beliefss)), inference[j][1]
		)
		posterior_distrib = [0.0] * 3  # only 3 possible goals
		for p in posterior_belief:  # doing this to preserve order in predictions
			inferred_lane_idx = possible_goals.index(p[0])  # going from lane to index for consistency
			inferred_lane_prob = p[1]  # what is the actual probability assigned to this belief
			posterior_distrib[inferred_lane_idx] = inferred_lane_prob

		result.append(posterior_distrib)

	return result


def car_action_inference_to_posterior_distrib(goal_inference, next_action_probs):
	result = []
	possible_actions = ["forward", "left", "right", "stop", "signal"]
	for i in range(len(next_action_probs)):
		try:
			next_action_prob = next_action_probs[i]
		except:
			pdb.set_trace()
		goal_distrib = goal_inference[i]
		log_res = np.zeros(len(possible_actions))
		for goal in next_action_prob:  # inferred actions given goal
			goal_prob = goal_distrib[possible_actions.index(goal)]  # probability of goal

			next_action_prob[goal] = np.clip(next_action_prob[goal], 1e-6, None)

			log_res += (goal_prob * np.log(next_action_prob[goal]))  # weigh action combination
		log_normalized_weights = log_res - scipy.special.logsumexp(log_res)
		res = np.exp(log_normalized_weights)
		result.append(list(res))
	return result




def get_car_L0_is_inference(rollout_env, states_raw, actions_raw, targetAgentID, carExistPrior=0.5, 
	num_samples=3, sampled_actions=10, lookAheadDepth=5, 
	full_action_history=[], state_belief_model=None, exist_belief_model=None):
	'''
	rollout_env is a Scenario object
	states_raw are the fully observable states of the world, and are a World object
	actions_raw is the action history of the agent you want to analyze
	'''

	# process partially observable states from perspective of OG agent if targetAgentID is in line of sight at beginning

	startTime = len(states_raw)  # this way, if an agent doesn't exist it'll just be uniform probs
	for t, s in enumerate(states_raw):
		if s.agent_exists(targetAgentID):
			startTime = t 
			break
	goal_inferences = []
	action_prob_inf = []
	for temp in range(startTime):
		goal_inferences.append([1/3] * 3)
		action_prob_inf.append([0.25, 0.25, 0.25, 0.25, 0.0])

	if startTime != len(states_raw):
		all_inferences = particle_inference(
			targetAgentID, 
			rollout_env, 
			states_raw[startTime:], 
			actions_raw[startTime:], 
			num_samples=num_samples,
			output_every_timestep=True,
			carExistPrior=carExistPrior,
			sampled_actions=sampled_actions, 
			lookAheadDepth=lookAheadDepth,
			full_action_history=full_action_history[startTime:],
			state_belief_model=state_belief_model,
			exist_belief_model=exist_belief_model
		)
		goal_inference, action_probs = all_inferences[0], all_inferences[1]
		# assert len(goal_inference) == len(states_raw)
		IS_goal_inferences = car_goal_inference_to_posterior_distrib(goal_inference)
		IS_action_inferences = car_action_inference_to_posterior_distrib(IS_goal_inferences, action_probs)
		goal_inferences += IS_goal_inferences
		action_prob_inf += IS_action_inferences
	return np.array(goal_inferences), np.array(action_prob_inf)
