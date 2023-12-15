from scenario import Scenario1, Action, get_partial_states
from drivingAgents.car_agent_L0 import AgentL0, locToGoalString
from drivingAgents.car_agent_L1 import AgentL1
import random
import numpy as np
import _pickle as pickle
import pdb
import time
import tqdm
import scipy.special
from scipy.special import softmax
from car_utils.car_sampler import sample_lane_utilities
from reasoning_about_car_L0 import car_goal_inference_to_posterior_distrib, car_action_inference_to_posterior_distrib

def save_tree_figs(save_dir, states_paths):
	from visualizer import Visualizer
	for state, f in states_paths:
		state.visualizer = Visualizer(state.width, state.height, ppm=state.ppm)
		state.render(save_dir, f)


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


def init_particles(env, data, num_samples=3, lane_utilities_proposal_probs=None, carExistPrior=0.5, beta_L0=0.001, beta_L1=0.001, 
	sampled_actions=10, lookAheadDepth=5, history=None, other_agent_inference_algorithm="IS", other_agent_num_samples=3,
	L0_inference_model=None, signal_danger_prior=0.5, state_model=None, exist_model=None, randomChoice=False):
	'''
	Assuming the state extracted from the data is a tuple of 
		(fully observable World object, action dict for all agents from that world)
	'''
	action_space = list(Action)
	state, L1_action, agentID = data
	action_idx = action_space.index(Action(L1_action.value))
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
			if sample_id < len(distinct_lanes) and num_samples >= num_possible_lane_targets and not randomChoice:
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

		emptyVisitedParticles = {}
		for i in range(16):
			emptyVisitedParticles[i] = {}

		clonedAgent = AgentL1(
			agentID,
			lane_utilities,
			pickle.loads(pickle.dumps(env.initialLocMapping)),
			pickle.loads(pickle.dumps(env)),
			env.transition,
			inference_algorithm=other_agent_inference_algorithm,
			signalDangerWeight=signal_danger_prior,
			car_exists_prior = carExistPrior,
			beta_L0=beta_L0,
			beta_L1=beta_L1,
			num_samples=other_agent_num_samples,
			lookAheadDepth=lookAheadDepth,
			model=L0_inference_model,
			visitedParticles=emptyVisitedParticles,
			state_model=state_model,
			exist_model=exist_model
		)

		agent_clones.append(clonedAgent)

		#env.reset()
		# pdb.set_trace()
		partialObs = get_partial_states(state[0], id=agentID)  # get the partial observation just for this agent
		agent_clones[sample_id].partial_obs.append((partialObs, state[1]))  # observation is partial obs and full action dict
		agent_clones[sample_id].personal_L0.observations.append(partialObs)
		agent_clones[sample_id].personal_L0.full_action_history.append(state[1])

		# WEIGH based on p(a_1 | ...)p(s | ...)
		# Score p(a_1 | ...)
		action_p, belief = agent_clones[sample_id].get_action_probs()
		action_prob = action_p[action_idx]

		beliefss[sample_id].append(belief)  # belief is a dictionary of beliefs about other agents

		# Assign the weight
		# if lane_utilities_proposal_probs is None or True:  # ignore the else statement for now
		if action_prob < 1e-6:
			log_weight[sample_id] = -1e6
		else:
			log_weight[sample_id] = np.log(action_prob)


		observableAgentIDs = []

		for otherCar in partialObs.dynamic_agents:
			if otherCar != agentID:
				observableAgentIDs.append(otherCar.ID)

		# temporary stepping ahead one timestep
		temp_action_dict = {}
		for car in partialObs.dynamic_agents: 
			if car.ID == agentID:
				continue
			elif car.ID not in observableAgentIDs or not partialObs.agent_exists(car.ID):  # one we imagine to exist, so it keeps prev action
				temp_action_dict[car.ID] = car.prev_action
			else:
				intentionBelief = belief[car.ID]  # what do we belief about its goal
				action_probabilities = [1e-6] * 5  # L1 has 5 actions
				L0_agent_clones = agent_clones[sample_id].particles[car.ID][2]

				for L0_agent_clone in L0_agent_clones:
					goal = L0_agent_clone.goalActionString
					goal_prob = intentionBelief[goal]
					L0_potential_action, L0_potential_action_info = L0_agent_clone.get_action(partialObs, return_info=True, prev_actions=state[1])  # get action probs
					L0_potential_action_probs = L0_potential_action_info['action_probs']
					for i, a_prob in enumerate(L0_potential_action_probs):  # weigh overall L0 action by action probs
						action_probabilities[i] += goal_prob * a_prob
					L0_agent_clone.observations.pop()  # remove curr_state from imagined future so we don't impact the gt
					L0_agent_clone.full_action_history.pop()
				log_probs = np.log(action_probabilities)

				normalized_log_probs = log_probs - scipy.special.logsumexp(log_probs)
				action_probabilities = list(softmax(normalized_log_probs))
				
				max_actions = [a for i, a in enumerate(action_space[:4]) if action_probabilities[i] == max(action_probabilities)]
				random.shuffle(max_actions)  # avoiding uniform being stuck
				temp_action_dict[car.ID] = random.choice(max_actions)

		temp_action_dict[agentID] = L1_action

		next_partial_obs = agent_clones[sample_id].transition(partialObs, temp_action_dict)

		# build belief so you can do inference over next action probabilities
		agent_clones[sample_id].partial_obs.append((next_partial_obs, temp_action_dict))
		agent_clones[sample_id].personal_L0.observations.append(next_partial_obs)
		agent_clones[sample_id].personal_L0.full_action_history.append(temp_action_dict)
		next_action_prob, next_belief = agent_clones[sample_id].get_action_probs()
		next_action_probs[lane_utilities] = next_action_prob  # store 

		next_partial_obs = agent_clones[sample_id].partial_obs.pop()
		agent_clones[sample_id].personal_L0.observations.pop()
		agent_clones[sample_id].personal_L0.full_action_history.pop()
		del next_partial_obs
		del temp_action_dict


	# Weight correction
	log_weight, too_low = get_corrected_log_weight(log_weight)

	particles = lane_utilitiess, beliefss, agent_clones
	return particles, log_weight, next_action_probs



def update_particles(particles, log_weight, data, lane_utilities_proposal_probs=None):
	'''
	Assuming the state extracted from the data is the fully observable Scenario.w World object
	'''
	action_space = list(Action)
	try:
		num_samples = len(log_weight)
	except:
		pdb.set_trace()
	lane_utilitiess, beliefss, agent_clones = particles
	prev_state, prev_L1_action, state, L1_action, agentID = data
	action_idx = action_space.index(Action(L1_action.value))
	updated_log_weight = np.zeros((num_samples,))
	num_timesteps = len(beliefss[0])


	next_action_probs = {}

	for sample_id in range(num_samples):
		partialObs = get_partial_states(state[0], id=agentID)  # get the partial observation just for this agent
		notVisible = False
		if type(partialObs) == list:  # temporarily append previous belief
			partialObs = agent_clones[sample_id].partial_obs[-1][0]
			notVisible = True
		if not notVisible:  # only save observations we actually see or those artifically sampled ahead
			agent_clones[sample_id].partial_obs.append((partialObs, state[1]))  # observation is partial obs and full action dict
			agent_clones[sample_id].personal_L0.observations.append(partialObs)
			agent_clones[sample_id].personal_L0.full_action_history.append(state[1])
		
		# WEIGH based on p(a_1 | ...)p(s | ...)
		# Score p(a_1 | ...)
		if notVisible:  # if we can't see the agent copy the goal inference from previous time
			updated_log_weight[sample_id] = log_weight[sample_id]  
			belief = beliefss[sample_id][-1]
		else: # if we are able to see the agent, update belief
			action_p, belief = agent_clones[sample_id].get_action_probs()
			action_prob = action_p[action_idx]

			# Assign the weight
			if action_prob < 1e-6:
				log_action_prob = -1e6
			else:
				log_action_prob = np.log(action_prob)
			alpha = 0.999
			p_uniform = 1 / num_samples
			updated_log_weight[sample_id] = log_weight[sample_id] + log_action_prob
			#updated_log_weight[sample_id] = np.log(alpha * np.exp(log_weight[sample_id]) + (1 - alpha) * p_uniform) + log_action_prob

		beliefss[sample_id].append(belief)

		observableAgentIDs = []
		for otherCar in partialObs.dynamic_agents:
			if otherCar != agentID:
				observableAgentIDs.append(otherCar.ID)



		# temporary stepping ahead one timestep
		temp_action_dict = {}
		for car in partialObs.dynamic_agents: 
			if car.ID == agentID:
				continue
			elif car.ID not in observableAgentIDs or not partialObs.agent_exists(car.ID):  # one we imagine to exist, so it keeps prev action
				temp_action_dict[car.ID] = car.prev_action
			else:
				intentionBelief = belief[car.ID]  # what do we belief about its goal
				action_probabilities = [1e-6] * 5  # L1 has 5 actions
				L0_agent_clones = agent_clones[sample_id].particles[car.ID][2]

				for L0_agent_clone in L0_agent_clones:
					goal = L0_agent_clone.goalActionString
					goal_prob = intentionBelief[goal]
					L0_potential_action, L0_potential_action_info = L0_agent_clone.get_action(partialObs, return_info=True, prev_actions=state[1])  # get action probs
					L0_potential_action_probs = L0_potential_action_info['action_probs']
					for i, a_prob in enumerate(L0_potential_action_probs):  # weigh overall L0 action by action probs
						action_probabilities[i] += goal_prob * a_prob
					L0_agent_clone.observations.pop()  # remove curr_state from imagined future so we don't impact the gt
					L0_agent_clone.full_action_history.pop()
				log_probs = np.log(action_probabilities)

				normalized_log_probs = log_probs - scipy.special.logsumexp(log_probs)
				action_probabilities = list(softmax(normalized_log_probs))
				
				max_actions = [a for i, a in enumerate(action_space[:4]) if action_probabilities[i] == max(action_probabilities)]
				random.shuffle(max_actions)  # avoiding uniform being stuck
				temp_action_dict[car.ID] = random.choice(max_actions)

		temp_action_dict[agentID] = L1_action

		next_partial_obs = agent_clones[sample_id].transition(partialObs, temp_action_dict)

		# build belief so you can do inference over next action probabilities
		agent_clones[sample_id].partial_obs.append((next_partial_obs, temp_action_dict))
		agent_clones[sample_id].personal_L0.observations.append(next_partial_obs)
		agent_clones[sample_id].personal_L0.full_action_history.append(temp_action_dict)
		next_action_prob, next_belief = agent_clones[sample_id].get_action_probs()
		next_action_probs[lane_utilitiess[sample_id]] = next_action_prob  # store 
		if not notVisible:  # we want to artificially step ahead cars we can't see
			next_partial_obs = agent_clones[sample_id].partial_obs.pop()
			agent_clones[sample_id].personal_L0.observations.pop()
			agent_clones[sample_id].personal_L0.full_action_history.pop()
		# del next_partial_obs
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
	other_agent_num_samples=3,
	lane_utilities_proposal_probss=None, 
	output_every_timestep=False,
	visitedParticles=None,
	carExistPrior=0.5,
	sampled_actions=10, 
	lookAheadDepth=5,
	beta_L0=0.001,
	beta_L1=0.001,
	other_agent_inference_algorithm="IS",
	L0_inference_model=None,
	signal_danger_prior=0.5,
	state_model=None,
	exist_model=None, 
	randomChoice=False
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


	if lane_utilities_proposal_probss is None:
		lane_utilities_proposal_probs = None 
	else:
		lane_utilities_proposal_probs = lane_utilities_proposal_probss[0]
	(lane_utilitiess, beliefss, agent_clones), log_weights[:, 0], next_action_probs = init_particles(
		env,
		data=(states[0], actions[0], agentID),
		num_samples=num_samples,
		lane_utilities_proposal_probs=lane_utilities_proposal_probs,
		history=visitedParticles,
		carExistPrior=carExistPrior,
		sampled_actions=sampled_actions, 
		lookAheadDepth=lookAheadDepth,
		beta_L0=beta_L0,
		beta_L1=beta_L1,
		other_agent_inference_algorithm=other_agent_inference_algorithm,
		L0_inference_model=L0_inference_model,
		signal_danger_prior=signal_danger_prior,
		other_agent_num_samples=other_agent_num_samples,
		state_model=state_model,
		exist_model=exist_model,
		randomChoice=randomChoice
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
		
				(
					(temp_lane_util, temp_belief, temp_agent_clone),
					temp_log_weights, temp_next_action_probs
				) = update_particles(
					particles=([util], [beliefss[i]], [agent_clones[i]]),
					log_weight=[log_weights[i, timestep - 1]],
					data=(states[timestep - 1], actions[timestep - 1], states[timestep], actions[timestep], agentID),
					lane_utilities_proposal_probs=lane_utilities_proposal_probs,
				)
				# need for actual inference
				# we only index into 0 because it return an array in the form [[x]], so we remove the extra dimension
				lane_utilitiess[i] = temp_lane_util[0]
				beliefss[i] = temp_belief[0]
				agent_clones[i] = temp_agent_clone[0]
				log_weights[i, timestep] = temp_log_weights[0]
				if timestep not in next_action_weights:
					next_action_weights[timestep] = {}
				next_action_weights[timestep][lane_utilitiess[i]] = temp_next_action_probs[lane_utilitiess[i]]


				# store to avoid repeated computation
				visitedParticles[(desire_goal, timestep+1)] = (beliefss[i], agent_clones[i], log_weights[i, timestep], next_action_weights[timestep][lane_utilitiess[i]])


		if not output_every_timestep:
			return (lane_utilitiess, beliefss, agent_clones), log_weights[:, -1], next_action_weights
	
	for timestep in tqdm.tqdm(range(1, num_timesteps)):
		lane_utilities_proposal_probs = None
		(
			(lane_utilitiess, beliefss, agent_clones),
			log_weights[:, timestep], next_action_probs
		) = update_particles(
			particles=(lane_utilitiess, beliefss, agent_clones),
			log_weight=log_weights[:, timestep - 1],
			data=(states[timestep - 1], actions[timestep - 1], states[timestep], actions[timestep], agentID),
			lane_utilities_proposal_probs=lane_utilities_proposal_probs,
		)
		next_action_weights[timestep] = next_action_probs
		if output_every_timestep:
			result.append(((lane_utilitiess, beliefss, agent_clones), log_weights[:, timestep]))


	if output_every_timestep:
		return result, next_action_weights
	else:
		return (lane_utilitiess, beliefss, agent_clones), log_weights[:, -1], next_action_weights


def online_importance_sampling(
	agentID,
	env,
	states, 
	actions, 
	num_samples=3, 
	other_agent_num_samples=3,
	lane_utilities_proposal_probss=None, 
	output_every_timestep=True,
	visitedParticles=None,
	carExistPrior=0.5,
	sampled_actions=10, 
	lookAheadDepth=5,
	beta_L0=0.001,
	beta_L1=0.001,
	other_agent_inference_algorithm="IS",
	L0_inference_model=None,
	signal_danger_prior=0.5,
	state_model=None,
	exist_model=None,
	randomChoice=False,
	signalTime=None,
	saveSignalDir=None
):
	num_timesteps = len(states)
	num_possible_lane_targets = 3
	num_possible_actions = 5
	# Initialize weights
	log_weights = np.zeros((num_samples, num_timesteps))
	next_action_weights = {0: {}}

	# FIRST TIMESTEP
	if lane_utilities_proposal_probss is None:
		lane_utilies_probs = None
	else:
		lane_utilies_probs = lane_utilities_proposal_probss[0:]

	(
		(lane_utilitiess, beliefss, agent_clones),
		log_weights[:, 0], next_action_weights
	) = particle_inference(
		agentID,
		pickle.loads(pickle.dumps(env)),
		pickle.loads(pickle.dumps(states[:1])),
		pickle.loads(pickle.dumps(actions[:1])), 
		num_samples=num_samples, 
		other_agent_num_samples=other_agent_num_samples,
		lane_utilities_proposal_probss=lane_utilies_probs, 
		output_every_timestep=False,
		carExistPrior=carExistPrior,
		sampled_actions=sampled_actions, 
		lookAheadDepth=lookAheadDepth,
		beta_L0=beta_L0,
		beta_L1=beta_L1,
		other_agent_inference_algorithm=other_agent_inference_algorithm,
		L0_inference_model=L0_inference_model,
		signal_danger_prior=signal_danger_prior,
		state_model=state_model,
		exist_model=exist_model,
		randomChoice=randomChoice
	)
	
	result = [((lane_utilitiess, beliefss, agent_clones), log_weights[:, 0])]

	visitedParticles = {}
	shiftedTime = max(next_action_weights) + 1
	for i, util in enumerate(lane_utilitiess):
		infoToStore = (beliefss[i], agent_clones[i], log_weights[i, 0], next_action_weights[0][util])
		visitedParticles[(util, shiftedTime)] = infoToStore

	# NEXT TIMESTEPS
	for timestep in tqdm.tqdm(range(1, num_timesteps)):
		if lane_utilities_proposal_probss is None:
			lane_utilies_probs = None
		else:
			lane_utilies_probs = lane_utilities_proposal_probss[timestep:]

		(
			(lane_utilitiess, beliefss, agent_clones),
			log_weights[:, timestep], temp_next_action_weights
		) = particle_inference(
			agentID,
			pickle.loads(pickle.dumps(env)),
			pickle.loads(pickle.dumps(states[:timestep+1])),
			pickle.loads(pickle.dumps(actions[:timestep+1])), 
			num_samples=num_samples, 
			other_agent_num_samples=other_agent_num_samples,
			lane_utilities_proposal_probss=lane_utilies_probs, 
			output_every_timestep=False,
			carExistPrior=carExistPrior,
			sampled_actions=sampled_actions, 
			lookAheadDepth=lookAheadDepth,
			beta_L0=beta_L0,
			beta_L1=beta_L1,
			other_agent_inference_algorithm=other_agent_inference_algorithm,
			L0_inference_model=L0_inference_model,
			signal_danger_prior=signal_danger_prior,
			state_model=state_model,
			exist_model=exist_model,
			visitedParticles=visitedParticles,
			randomChoice=randomChoice
		)

		if signalTime is not None:
			if timestep != signalTime:
				pass 
			else:
				time = signalTime
				statesFilenames = [(states[time][0], "L2_base")]
				for i, lane in enumerate(lane_utilitiess):
					print(f"Log weight for goal {lane} L2 Agent {agentID}: {log_weights[i, time]}")
					print(f"----------------")

					
					# PHYSICAL STATE BELIEF OF MAIN GREEN AGENT
					L1_state_belief = agent_clones[i].personal_L0.get_belief()[0][0][0]
					L1_state_filename = f"car{agentID}_level1_{lane}"
					statesFilenames.append((L1_state_belief, L1_state_filename))

					for L0_agent, weights in agent_clones[i].log_weight_dict.items():
						if weights is not None:
							for j, w in enumerate(weights):
								subgoal = agent_clones[i].particles[L0_agent][0][j]
								print(f"    Log weight for goal {subgoal} L1 Agent {L0_agent}: {w}")
								print(f"    --------")
								state_beliefs = agent_clones[i].particles[L0_agent][1][j][time - 1]

								for state, s_prob in state_beliefs:
									L0_state_filename = f"car{L0_agent}_level0_{subgoal}"
									statesFilenames.append((state, L0_state_filename))

									print(f"        Probability of state for goal {subgoal} L0 Agent {L0_agent}: {s_prob}")
									print(f"        ----")
									for c in state.dynamic_agents:
										print(f"            Car {c.ID} (x, y) = ({c.center.x}, {c.center.y}); angle: {c.heading} speed: {c.speed}")
								print("\n")	
							print("\n")	
					print("\n")	
				save_tree_figs(saveSignalDir, statesFilenames)



		for k, actionProbDict in temp_next_action_weights.items():
			if k not in next_action_weights:  # if the timestep doesn't exist, make an entry
				next_action_weights[k] = actionProbDict
			else:  # if the timestep and util already exists, update it
				for u, actionProb in actionProbDict.items():
					next_action_weights[k][u] = actionProb

		# shiftedTime = max(next_action_weights) + 1
		# for i, util in enumerate(lane_utilitiess):
		# 	infoToStore = (beliefss[i], agent_clones[i], log_weights[i, timestep], next_action_weights[timestep][util])
		# 	visitedParticles[(util, shiftedTime+1)] = infoToStore
		result.append(((lane_utilitiess, beliefss, agent_clones), log_weights[:, timestep]))

	

				

	if output_every_timestep:
		return result, next_action_weights
	else:
		return (lane_utilitiess, beliefss, agent_clones), log_weights[:, -1], next_action_weights



def get_car_L1_is_inference(rollout_env, states_raw, actions_raw, targetAgentID, carExistPrior=0.65, 
	num_samples=3, sampled_actions=10, lookAheadDepth=3, other_agent_num_samples=3, beta_L0=0.001, beta_L1=0.001,
	other_agent_inference_algorithm="IS", L0_inference_model=None, signal_danger_prior=0.5,
	state_model=None,exist_model=None):
	'''
	rollout_env is a Scenario object
	states_raw is the tuple of (fully observable states of the world, full action dict)
	actions_raw is the action history of the agent you want to analyze as a list
	'''

	# process partially observable states from perspective of OG agent if targetAgentID is in line of sight at beginning

	startTime = len(states_raw)  # this way, if an agent doesn't exist it'll just be uniform probs
	for t, s in enumerate(states_raw):
		if s[0].agent_exists(targetAgentID):
			startTime = t 
			break
	goal_inferences = []
	action_prob_inf = []
	for temp in range(startTime):
		goal_inferences.append([1/3] * 3)
		action_prob_inf.append([0.2] * 5)

	if startTime != len(states_raw):
		all_inferences = particle_inference(
			targetAgentID,
			rollout_env,
			states_raw[startTime:], 
			actions_raw[startTime:], 
			num_samples=num_samples, 
			other_agent_num_samples=other_agent_num_samples,
			lane_utilities_proposal_probss=None, 
			output_every_timestep=True,
			visitedParticles=None,
			carExistPrior=carExistPrior,
			sampled_actions=sampled_actions, 
			lookAheadDepth=lookAheadDepth,
			beta_L0=beta_L0,
			beta_L1=beta_L1,
			other_agent_inference_algorithm=other_agent_inference_algorithm,
			L0_inference_model=L0_inference_model,
			signal_danger_prior=signal_danger_prior,
			state_model=state_model,
			exist_model=exist_model,
		)

		goal_inference, action_probs = all_inferences[0], all_inferences[1]
		# assert len(goal_inference) == len(states_raw)
		IS_goal_inferences = car_goal_inference_to_posterior_distrib(goal_inference)
		IS_action_inferences = car_action_inference_to_posterior_distrib(IS_goal_inferences, action_probs)
		goal_inferences += IS_goal_inferences
		action_prob_inf += IS_action_inferences
	return np.array(goal_inferences), np.array(action_prob_inf)


def get_L1_online_inference(rollout_env, states_raw, actions_raw, targetAgentID, carExistPrior=0.65, 
	num_samples=3, sampled_actions=10, lookAheadDepth=3, other_agent_num_samples=3, beta_L0=0.001, beta_L1=0.001,
	other_agent_inference_algorithm="IS", L0_inference_model=None, signal_danger_prior=0.5,
	state_model=None,exist_model=None, lane_utilities_proposal_probss=None, randomChoice=False, signalTime=None, saveSignalDir=None):
	'''
	rollout_env is a Scenario object
	states_raw is the tuple of (fully observable states of the world, full action dict)
	actions_raw is the action history of the agent you want to analyze as a list
	'''

	# process partially observable states from perspective of OG agent if targetAgentID is in line of sight at beginning

	startTime = len(states_raw)  # this way, if an agent doesn't exist it'll just be uniform probs
	for t, s in enumerate(states_raw):
		if s[0].agent_exists(targetAgentID):
			startTime = t 
			break
	goal_inferences = []
	action_prob_inf = []
	for temp in range(startTime):
		goal_inferences.append([1/3] * 3)
		action_prob_inf.append([0.2] * 5)

	if lane_utilities_proposal_probss is not None:
		lane_utilities_proposal_probs = lane_utilities_proposal_probss[startTime:]
	else:
		lane_utilities_proposal_probs = None

	if startTime != len(states_raw):
		all_inferences = online_importance_sampling(
			targetAgentID,
			rollout_env,
			states_raw[startTime:], 
			actions_raw[startTime:], 
			num_samples=num_samples, 
			other_agent_num_samples=other_agent_num_samples,
			lane_utilities_proposal_probss=lane_utilities_proposal_probs, 
			output_every_timestep=True,
			visitedParticles=None,
			carExistPrior=carExistPrior,
			sampled_actions=sampled_actions, 
			lookAheadDepth=lookAheadDepth,
			beta_L0=beta_L0,
			beta_L1=beta_L1,
			other_agent_inference_algorithm=other_agent_inference_algorithm,
			L0_inference_model=L0_inference_model,
			signal_danger_prior=signal_danger_prior,
			state_model=state_model,
			exist_model=exist_model,
			randomChoice=randomChoice,
			signalTime=signalTime,
			saveSignalDir=saveSignalDir
		)

		goal_inference, action_probs = all_inferences[0], all_inferences[1]
		# assert len(goal_inference) == len(states_raw)
		goal_distrib = car_goal_inference_to_posterior_distrib(goal_inference)
		action_distrib = car_action_inference_to_posterior_distrib(goal_distrib, action_probs)
		goal_inferences += goal_distrib
		action_prob_inf += action_distrib
	return np.array(goal_inferences), np.array(action_prob_inf)

