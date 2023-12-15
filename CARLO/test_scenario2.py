from scenario import Scenario1, Action, get_partial_states
import random
import numpy as np
import _pickle as pickle
from drivingAgents.car_agent_L0 import AgentL0
from drivingAgents.car_agent_L1 import AgentL1
import reasoning_about_car_L0
import reasoning_about_car_L1
from reasoning_about_car_L1 import get_L1_online_inference, get_car_L1_is_inference
import pdb
import time
import torch
from tqdm import tqdm
import car_utils.general as general
from car_utils.network import load_checkpoint
import shutil
from pathlib import Path
from car_utils.network import state_action_to_joint_tensor, agent_pair_to_tensor, load_belief_checkpoint
from agents import Car
import matplotlib.pyplot as plt
from scipy.special import rel_entr


def eliminate_zero(distribution):
	'''
	Adds small noise to eliminate zeroed out values and make kl divergence stable
	'''
	num_nonzeros = np.count_nonzero(distribution)
	if num_nonzeros == 0:  # everything is 0, so make uniform
		dist_length = len(distribution)
		distribution = [1/dist_length] * dist_length
	else:
		num_zeros = len(distribution) - num_nonzeros # find number of zeros 
		if num_zeros > 0:
			noise_subtraction = (1e-6 * num_zeros) / num_nonzeros  # take away uniformly from non-zeros so we sum to 1
			for j, prob in enumerate(distribution):
				if prob == 0:  # add some noise
					distribution[j] = 1e-6
				else:
					distribution[j] -= noise_subtraction

def KL(gt_distrib, approx_distrib):
	'''
	Calculates KL divergence between 1d arrays
	'''
	eliminate_zero(gt_distrib)
	eliminate_zero(approx_distrib)
	return sum(rel_entr(gt_distrib, approx_distrib))

def avg_KL(gt_distribs, approx_distribs):
	'''
	Calculates average KL divergence between 2d arrays
	'''
	return np.mean([KL(gt_distrib, approx_distribs[i]) for i, gt_distrib in enumerate(gt_distribs)])

def numberCorrect(predictions, target):
	'''
	Returns average number of instances in which max probability proposal was true utility
	'''
	if type(target) == list and type(target) != torch.Tensor:
		correct_classes = np.array(target)
	elif type(target) == torch.Tensor:
		correct_classes = target.cpu().detach().numpy()
	else:
		correct_classes = np.array([target] * len(predictions))

	if type(predictions) == list:
		predictions = np.array(predictions)
	if type(predictions) == torch.Tensor:
		predictions = predictions.cpu().detach().numpy()

	num_samples, num_classes = predictions.shape
	correct_count = 0

	for i in range(num_samples):
		ground_truth = correct_classes[i]
		max_probabilities = [prob for prob in predictions[i] if prob == max(predictions[i])]
		num_max_probs = len(max_probabilities)

		# Handle tie-breakers
		if ground_truth in [idx for idx, prob in enumerate(predictions[i]) if prob == max(predictions[i])]:
			# correct_count += 1 / num_max_probs
			correct_count += 1

	accuracy = correct_count / num_samples
	return accuracy

def correctSignalPct(predictions, target):
	if type(target) == list and type(target) != torch.Tensor:
		correct_classes = np.array(target)
	elif type(target) == torch.Tensor:
		correct_classes = target.cpu().detach().numpy()
	else:
		correct_classes = np.array([target] * len(predictions))

	if type(predictions) == list:
		predictions = np.array(predictions)
	if type(predictions) == torch.Tensor:
		predictions = predictions.cpu().detach().numpy()



	num_samples, num_classes = predictions.shape

	actionTimes = [x for (x, ac) in enumerate(target) if ac == 4]  # only use relevant timesteps
	num_signals = len(actionTimes)
	if num_signals == 0:
		return None

	correct_count = 0

	for i in range(num_samples):
		if i not in actionTimes:
			continue
		ground_truth = correct_classes[i]
		max_probabilities = [prob for prob in predictions[i] if prob == max(predictions[i])]
		num_max_probs = len(max_probabilities)

		# Handle tie-breakers
		if ground_truth in [idx for idx, prob in enumerate(predictions[i]) if prob == max(predictions[i])]:
			correct_count += 1 / num_max_probs

	accuracy = correct_count / num_signals
	return accuracy




def make_car_gif(gif_path, save_dir, states_raw):
	from visualizer import Visualizer
	tmp_dir = general.get_tmp_dir()
	img_paths  = []
	for timestep, state in enumerate(states_raw):
		state.visualizer = Visualizer(state.width, state.height, ppm=state.ppm)
		state.render(save_dir + tmp_dir, timestep)
		img_paths.append(f"{save_dir}{tmp_dir}/{timestep}_gui.png")

	general.make_gif(img_paths, save_dir + gif_path, 3)
	shutil.rmtree(save_dir + tmp_dir)

def compare_values(a, b, tol=1e-8):
	# Custom comparison function to handle tie-breaking based on tolerance
	return abs(a - b) <= tol


def percentCorrect(predictions, target):
	'''
	Returns average number of instances in which max probability proposal was true utility
	'''
	
	if type(predictions) == torch.Tensor:
		predictions = predictions.cpu().detach().numpy()
	if type(target) == torch.Tensor:
		correct_classes = target.cpu().detach().numpy()

	num_samples = predictions.shape[0]
	max_prob_indices = np.argmax(predictions, axis=1)
	max_probs = predictions[np.arange(num_samples), max_prob_indices]
	ground_truth_probs = predictions[np.arange(num_samples), correct_classes]

	# Handle tie-breakers
	is_tie = np.sum(predictions == np.max(predictions, axis=1, keepdims=True), axis=1) > 1
	correct_count = np.sum((ground_truth_probs >= max_probs) & is_tie) + np.sum(ground_truth_probs == max_probs)

	accuracy = correct_count / num_samples
	return accuracy

def eliminate_zero(distribution):
	'''
	Adds small noise to eliminate zeroed out values and make kl divergence stable
	'''
	num_nonzeros = np.count_nonzero(distribution)
	if num_nonzeros == 0:  # everything is 0, so make uniform
		dist_length = len(distribution)
		for j in range(len(distribution)):
			distribution[j] = 1 / dist_length
	else:
		num_zeros = len(distribution) - num_nonzeros # find number of zeros 
		if num_zeros > 0:
			noise_subtraction = (1e-6 * num_zeros) / num_nonzeros  # take away uniformly from non-zeros so we sum to 1
			for j, prob in enumerate(distribution):
				if prob == 0:  # add some noise
					distribution[j] = 1e-6
				else:
					distribution[j] -= noise_subtraction

def KL(gt_distrib, approx_distrib):
	'''
	Calculates KL divergence between 1d arrays
	'''
	eliminate_zero(gt_distrib)
	eliminate_zero(approx_distrib)
	return sum(rel_entr(gt_distrib, approx_distrib))

def avg_KL(gt_distribs, approx_distribs):
	'''
	Calculates average KL divergence between 2d arrays
	'''
	totKL = 0
	for i, gt_distrib in enumerate(gt_distribs):
		approx_distrib = approx_distribs[i]
		totKL += KL(gt_distrib, approx_distrib)
	return totKL / len(gt_distribs)


def generate_scenario2_rollout(car1_exist_prior=0.5, car2_exist_prior=0.5, sampled_actions=10, lookAheadDepth=5, gif_dir=None,
	num_samples=3, max_timesteps=30, other_agent_inference_algorithm="IS", 
	other_agent_inference_model=None, state_model=None, exist_model=None, generate_random=False, num_cars=3):
	# Init the scenario
	scenario = Scenario1(num_cars=num_cars)
	scenario.reset()

	idToCar = {}
	idToAgent = {}
 
	for car_num, car in enumerate(scenario.w.dynamic_agents):
		emptyVisitedParticles = {}
		for i in range(16):
			emptyVisitedParticles[i] = {}

		if (car_num == len(scenario.w.dynamic_agents) - 1) and (generate_random):  # make the last agent act randomly
			agent = AgentL1(
				car.ID,
				car.actionGoal.value,
				scenario.initialLocMapping,
				scenario,
				scenario.transition,
				inference_algorithm='random',
				signalDangerWeight=0.0,
				car_exists_prior = 0,
				beta_L0=0.01,
				beta_L1=0.01,
				num_samples=1,
				lookAheadDepth=1,
				model=other_agent_inference_model,
				visitedParticles=emptyVisitedParticles,
				exist_model = exist_model,
				state_model = state_model,
			)
		else:
			agent = AgentL1(
				car.ID,
				car.actionGoal.value,
				scenario.initialLocMapping,
				scenario,
				scenario.transition,
				inference_algorithm=other_agent_inference_algorithm,
				signalDangerWeight=0.5,
				car_exists_prior = car1_exist_prior,
				beta_L0=0.01,
				beta_L1=0.01,
				num_samples=num_samples,
				lookAheadDepth=lookAheadDepth,
				model=other_agent_inference_model,
				visitedParticles=emptyVisitedParticles,
				exist_model = exist_model,
				state_model = state_model,
			)


		idToCar[car.ID] = car 
		idToAgent[car.ID] = agent


	states = None
	full_observed_state_history = []  # third person perspective
	action_history = {}
	ground_truth_goals = {}  # what do both agents actually want to do
	prev_actions = {}
	for cID in idToAgent:
		action_history[cID] = [Action.FORWARD]
		ground_truth_goals[cID] = idToAgent[cID].goalActionString
		prev_actions[cID] = Action.FORWARD

	done = False
	prev_action_dict = None
	action_probs = []
	for step_number in tqdm(range(max_timesteps)):		
		grab_belief = False

		action_dict = {}
		action_info_dict = {}
		if states is None:
			for cID in idToAgent:  # all agents move forward initially
				action_dict[cID] = Action.FORWARD

		else:
			for cID in idToAgent:
				c_state = states[cID]
				c_observation = c_state[2][1] # 3rd element of state is partial observation - (agentID, obs)
				
				for other_cars in c_observation.dynamic_agents: # we do this so our agent imagines other agents keep maintaining their actions
					if other_cars.ID in prev_actions:
						other_cars.prev_action = prev_actions[other_cars.ID]

				c_action, c_info = idToAgent[cID].get_action(observation=(c_observation, prev_action_dict), return_info=True)
				action_info_dict[cID] = c_info["action_probs"]
				action_dict[cID] = c_action
				action_history[cID].append(c_action)

		full_observed_state_history.append((scenario.w.clone(), pickle.loads(pickle.dumps(action_dict))))
		next_states, reward, done, info = scenario.step(action_dict)

		prev_action_dict = pickle.loads(pickle.dumps(action_dict))
		states = next_states

		prev_actions = pickle.loads(pickle.dumps(action_dict))

		if len(action_info_dict) > 0:
			action_probs.append(action_info_dict)

		if done:
			break
	return scenario, full_observed_state_history, action_history, ground_truth_goals, done, action_probs

	# shutil.rmtree(save_dir + tmp_dir)
	

if __name__ == "__main__":
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(123)
	cuda = torch.cuda.is_available()
	if cuda:
		torch.cuda.manual_seed(123)
		device = "cuda"
	else:
		device = "cpu"

	BIG_STORAGE_DIR = ""
	# BIG_STORAGE_DIR = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning/CARLO/"
	

	L0_model_save_path = f"{BIG_STORAGE_DIR}save/scenario1/debug/num_sampled_actions=1,lookAheadDepth=1,beta=0.01/checkpoints/best_acc.pik"
	L0_inference_model, L0_optimizer, L0_stats, L0_args = load_checkpoint(L0_model_save_path, device)
	L0_inference_model.eval()

	state_belief_save_dir = f"{BIG_STORAGE_DIR}save/stateEstimation/debug/num_sampled_actions=1,lookAheadDepth=1,beta=0.01/checkpoints/"
	state_model_path = state_belief_save_dir + "best_acc_state.pik"
	exist_model_path = state_belief_save_dir + "best_acc_exist.pik"
	state_model, _, _, _ = load_belief_checkpoint(state_model_path, device, exist_model=False)
	exist_model, _, _, _ = load_belief_checkpoint(exist_model_path, device, exist_model=True)
	state_model.eval()
	exist_model.eval()

	L1_model_save_dir = f"{BIG_STORAGE_DIR}save/scenario2/debug/num_sampled_actions=1,lookAheadDepth=10,beta=0.01"
	
	model_dir_list = [L1_model_save_dir]
	# for i in range(5):
	# for i in [2, 5, 10]:
	# for i in [1, 2, 3]:
	# 	# model_dir_list.append(f"{BIG_STORAGE_DIR}save/scenario2/KLLess_{i}_pct_3.0kDat_128dim_0.0001lr_64bSize/num_sampled_actions=1,lookAheadDepth=10,beta=0.01")
	# 	model_dir_list.append(f"{BIG_STORAGE_DIR}save/scenario2/KLLess_{i}_6_3.0kDat_128dim_0.0001lr_64bSize/num_sampled_actions=1,lookAheadDepth=10,beta=0.01")

	# max_acc = []
	# for L1_model_save_dir in model_dir_list:
	# 	L1_model_save_path = f"{L1_model_save_dir}/checkpoints/best_acc.pik"
	# 	L1_inference_model, L1_optimizer, L1_stats, L1_args = load_checkpoint(L1_model_save_path, device)
	# 	L1_inference_model.eval()
	# 	max_acc.append(L1_stats['test_losses'][L1_stats['test_accuracies'].index(max(L1_stats['test_accuracies']))])

	L1_model_save_dir = model_dir_list[0]

	L1_model_save_path = f"{L1_model_save_dir}/checkpoints/best_acc.pik"
	# L1_model_save_path = f"{L1_model_save_dir}/checkpoints/epoch_50.pik"
	L1_inference_model, L1_optimizer, L1_stats, L1_args = load_checkpoint(L1_model_save_path, device)
	L1_inference_model.eval()

	L1_nll_model_dir = f"{BIG_STORAGE_DIR}save/scenario2/debug_nll_goals/num_sampled_actions=1,lookAheadDepth=10,beta=0.01"
	L1_nll_model_path = f"{L1_model_save_dir}/checkpoints/best_acc.pik"
	cross_ent_model, cross_ent_optimizer, cross_ent_stats, cross_ent_args = load_checkpoint(L1_nll_model_path, device, L1=True)
	cross_ent_model.eval()

	L1_nll_model_action_dir  = f"{BIG_STORAGE_DIR}save/scenario2/debug_nll_actions/num_sampled_actions=1,lookAheadDepth=10,beta=0.01"
	L1_nll_model_action_path  = f"{L1_nll_model_action_dir}/checkpoints/best_acc.pik"
	cross_ent_action_model, _, _, _ = load_checkpoint(L1_nll_model_action_path, device, L1=True, actionPred=True)
	cross_ent_action_model.eval()


	generate_random = False
	if not generate_random:
		saveExtension = ""
	else:
		saveExtension = "Rando"

	num_inference_samples_L1 = 3
	num_inference_samples_L0 = 3
	beta_L0 = beta_L1 = 0.01
	car_exist_prior = 0.65
	sampled_actions = 20
	lookAheadDepth = 5
	max_timesteps = 6
	num_trials = 2
	num_cars = 3
	plotting = False

	nnSignalAccuracy = []
	oursSignalAccuracy = []


	avgAccuracyPctTime = {}
	avgAccuracyPctTime['is'] = {i: [] for i in range(10)}
	avgAccuracyPctTime['nn'] = {i: [] for i in range(10)}
	action_kl = {'nn': []}
	action_kl['uniform'] = []   # uniform probs
	action_accuracy = {'is':[]}
	action_accuracy['nn'] = []
	for n in range(1, 4):
		for m in range(1, 4):
			if plotting and (n, m) != (3,3):
				continue
			action_accuracy[(n,m)] = []
			action_kl[(n,m)] = []
			avgAccuracyPctTime[(n,m)] = {i: [] for i in range(10)}
			action_accuracy[('random', n*m)] = []
			action_kl[('random', n*m)] = []
			avgAccuracyPctTime[('random', n*m)] = {i: [] for i in range(10)}


	is_nn_accuracy = 0.0
	nn_accuracy = 0.0

	for trial in tqdm(range(num_trials)):
		print(f"Batch {trial}")

		rollout = generate_scenario2_rollout(car1_exist_prior=car_exist_prior, car2_exist_prior=car_exist_prior, sampled_actions=10, lookAheadDepth=lookAheadDepth, 
			gif_dir=None, num_samples=num_inference_samples_L1, max_timesteps=max_timesteps, other_agent_inference_algorithm="Online_IS+NN", 
			other_agent_inference_model=L0_inference_model, state_model=state_model, exist_model=exist_model, generate_random=generate_random, num_cars=num_cars)		
		
		scenario, full_observed_state_history, action_history, ground_truth_goals, done, action_info_dict = rollout
		noSignal = True
		signalingAgent = None
		signalingTime = None
		for a, hist in action_history.items():
			for time, v in enumerate(hist):
				if Action.SIGNAL.value == v.value:
					noSignal = False
					signalingAgent = a
					signalingTime = time - 4
					break
			break  # we only care about the first agent
		if noSignal and plotting:
			continue

		ids = list(action_history.keys())

		goalInferenceCar1, goalInferenceCar2, goalInferenceCar3 = None, None, None
		actionInferenceCar1, actionInferenceCar2, actionInferenceCar3 = None, None, None
		car1_goal_idx, car2_goal_idx, car3_goal_idx = None, None, None
		partial_obs_history = None
		agent_id = None


		for car_number, id in enumerate(ids):
			if partial_obs_history is None:  # inferring from first car perspective
				partial_obs_history = [s[0] for s in full_observed_state_history]
				agent_id = id

			partial_obs_history = partial_obs_history[4:]  # clipping earlier sequence

			car_actions = action_history[id][4:]  # clipping earlier sequence
			car_goal = ground_truth_goals[id]
			car_goal_idx = reasoning_about_car_L0.GOAL_SPACE.index(car_goal)
			action_vals = list(Action)
			car_actions_ints = [action_vals.index(Action(a.value)) for a in car_actions]
			def get_partial_actions(time):
				partial_actions = pickle.loads(pickle.dumps(action_history))
				for a in partial_actions:  # get actions for every agent at time i
					partial_actions[a] = partial_actions[a][time]
				return partial_actions
			# clipping earlier sequence
			state_action_tensors = torch.stack(
				[
					state_action_to_joint_tensor(state_raw, get_partial_actions(i+4), device)
					for i, state_raw in enumerate(partial_obs_history)
				],
				dim=0,
			)
			state_actions = [state_action_tensors]

			id_tensors = [agent_pair_to_tensor(agent_id, id, device)]
			lens = torch.LongTensor([s.shape[0] for s in state_actions]).cpu()

			'''
			Baseline 1 -> Cross Ent nn baseline
			'''
			cross_ent_log_prob = cross_ent_action_model(state_actions, id_tensors, lens)
			cross_ent_nn_inferences = torch.softmax(cross_ent_log_prob, 1).cpu().detach().numpy()
			cross_ent_nn_inferences /= cross_ent_nn_inferences.sum(axis=1, keepdims=True)
			
			partial_state_action_tuple = list(zip(partial_obs_history, [s[1] for s in full_observed_state_history[4:]]))

			
			uniform_probs = np.full_like(cross_ent_nn_inferences, 1/5)


			'''
			Baseline 2 -> Exact Inference
			'''

			try:
				full_IS_goal_inferences, full_IS_action_inferences = get_car_L1_is_inference(scenario, 
						partial_state_action_tuple, car_actions, id, carExistPrior=car_exist_prior, 
						num_samples=3, sampled_actions=10, lookAheadDepth=lookAheadDepth, 
						other_agent_num_samples=3, beta_L0=beta_L0, beta_L1=beta_L1,
						other_agent_inference_algorithm="Online_IS+NN", L0_inference_model=L0_inference_model, signal_danger_prior=0.5,
						state_model=state_model, exist_model=exist_model)
			except:
				continue



			temp_accuracy = numberCorrect(cross_ent_nn_inferences, car_actions_ints)
			temp_signal_acc = correctSignalPct(cross_ent_nn_inferences, car_actions_ints)
			if temp_signal_acc is not None:
				nnSignalAccuracy.append(temp_signal_acc)


			temp_kl = avg_KL(full_IS_action_inferences, cross_ent_nn_inferences)
			action_kl['nn'].append(temp_kl)
			action_accuracy['nn'].append(temp_accuracy)

			temp_kl = avg_KL(full_IS_action_inferences, uniform_probs)
			action_kl['uniform'].append(temp_kl)

			# temp_accuracy = numberCorrect(full_IS_goal_inferences, goal_int)
			temp_accuracy = numberCorrect(full_IS_action_inferences, car_actions_ints)
			action_accuracy["is"].append(temp_accuracy)

			for t in range(1, len(full_IS_action_inferences) + 1):
				for pct in range(1, 11):
					if (t <= pct * len(full_IS_action_inferences) / 10):
						# avgAccuracyPctTime[key][pct-1].append(numberCorrect(inf[t-1:t], goal_int))
						avgAccuracyPctTime['is'][pct-1].append(numberCorrect(full_IS_action_inferences[t-1:t], car_actions_ints[t-1:t]))
						break

			for t in range(1, len(cross_ent_nn_inferences) + 1):
				for pct in range(1, 11):
					if (t <= pct * len(cross_ent_nn_inferences) / 10):
						# avgAccuracyPctTime[key][pct-1].append(numberCorrect(inf[t-1:t], goal_int))
						avgAccuracyPctTime['nn'][pct-1].append(numberCorrect(cross_ent_nn_inferences[t-1:t], car_actions_ints[t-1:t]))
						break

			''' 
			Baseline 3 -> Our Approach
			'''
			# our kl div nn model
			log_prob = L1_inference_model(state_actions, id_tensors, lens)
			nn_inferences = torch.softmax(log_prob, 1).cpu().detach().numpy()
			nn_inferences /= nn_inferences.sum(axis=1, keepdims=True)

			for n in range(1, 4):
				for m in range(1, 4):
					if plotting and (n, m) != (3,3):
						continue
					# if (n, m) != (3,3):
					# 	continue
					saveSignalDir= f"save/examples/tree{trial}/"
					try:
						goalInferenceCar, actionInferenceCar = get_L1_online_inference(scenario, 
								pickle.loads(pickle.dumps(partial_state_action_tuple)), car_actions, id, carExistPrior=car_exist_prior, 
								num_samples=n, sampled_actions=10, lookAheadDepth=lookAheadDepth, 
								other_agent_num_samples=m, beta_L0=beta_L0, beta_L1=beta_L1, 
								other_agent_inference_algorithm="Online_IS+NN", 
								L0_inference_model=L0_inference_model, signal_danger_prior=0.5, 
								state_model=state_model, exist_model=exist_model, 
								lane_utilities_proposal_probss=nn_inferences, signalTime=signalingTime, saveSignalDir=saveSignalDir)
					except:
						goalInferenceCar, actionInferenceCar = get_L1_online_inference(scenario, 
								pickle.loads(pickle.dumps(partial_state_action_tuple)), car_actions, id, carExistPrior=car_exist_prior, 
								num_samples=n, sampled_actions=10, lookAheadDepth=lookAheadDepth, 
								other_agent_num_samples=m, beta_L0=beta_L0, beta_L1=beta_L1, 
								other_agent_inference_algorithm="Online_IS+NN", 
								L0_inference_model=L0_inference_model, signal_danger_prior=0.5, 
								state_model=state_model, exist_model=exist_model, 
								lane_utilities_proposal_probss=nn_inferences)

					if not plotting:
						temp_accuracy = numberCorrect(actionInferenceCar, car_actions_ints)
						action_accuracy[(n,m)].append(temp_accuracy)
						
						if (n,m) == (3,3):
							temp_signal_acc = correctSignalPct(actionInferenceCar, car_actions_ints)
							if temp_signal_acc is not None:
								oursSignalAccuracy.append(temp_signal_acc)

						temp_kl = avg_KL(full_IS_action_inferences, actionInferenceCar)
						action_kl[(n,m)].append(temp_kl)
						for t in range(1, len(actionInferenceCar) + 1):
							for pct in range(1, 11):
								if (t <= pct * len(actionInferenceCar) / 10):
									# avgAccuracyPctTime[key][pct-1].append(numberCorrect(inf[t-1:t], goal_int))
									avgAccuracyPctTime[(n,m)][pct-1].append(numberCorrect(actionInferenceCar[t-1:t], car_actions_ints[t-1:t]))
									break

			for n in range(1, 4):
				for m in range(1, 4):
					if plotting:
						continue
					goalInferenceCar, actionInferenceCar = get_L1_online_inference(scenario, 
							partial_state_action_tuple, car_actions, id, carExistPrior=car_exist_prior, 
							num_samples=n, sampled_actions=10, lookAheadDepth=lookAheadDepth, 
							other_agent_num_samples=m, beta_L0=beta_L0, beta_L1=beta_L1, 
							other_agent_inference_algorithm="random", 
							L0_inference_model=L0_inference_model, signal_danger_prior=0.5, 
							state_model=state_model, exist_model=exist_model, 
							lane_utilities_proposal_probss=None,
							randomChoice=True)

					temp_accuracy = numberCorrect(actionInferenceCar, car_actions_ints)
					action_accuracy[('random', n*m)].append(temp_accuracy)
					
					temp_kl = avg_KL(full_IS_action_inferences, actionInferenceCar)
					action_kl[('random', n*m)].append(temp_kl)

					for t in range(1, len(actionInferenceCar) + 1):
						for pct in range(1, 11):
							if (t <= pct * len(actionInferenceCar) / 10):
								# avgAccuracyPctTime[key][pct-1].append(numberCorrect(inf[t-1:t], goal_int))
								avgAccuracyPctTime[('random', n*m)][pct-1].append(numberCorrect(actionInferenceCar[t-1:t], car_actions_ints[t-1:t]))
								break



			if plotting:

				path = f'save/finalResults/carRollouts/specificAccuracy{trial}_Signal.png'

				actualStates = []
				for state in partial_state_action_tuple:
					s = state[0].clone()
					s.dynamic_agents = []
					x = 'blue'
					for t in state[0].dynamic_agents:
						if t.ID == ids[0]:
							x = 'green'
						elif t.ID == ids[1]:
							x = 'red'
						else:
							x = 'blue'
			 
						s.add(Car(t.center, t.heading, ID=t.ID, color=x))

					actualStates.append(s)
				make_car_gif(f"specificAccuracy{trial}_Signal.gif", "save/finalResults/carRollouts/", actualStates)
				# want to generate plot of accuracy metrics vs time
				num_cols = 1
				num_rows = 1
				fig, axss = plt.subplots(
					num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), squeeze=False
				)
				axs = axss.flatten()

				key = (3, 2)
				inf = actionInferenceCar
				#inf = cross_ent_nn_inferences
				timestepLabels = list(range(len(inf)))

				forwardProbs = [x[0] for x in inf]
				leftProbs = [x[1] for x in inf]
				rightProbs = [x[2] for x in inf]
				stopProbs = [x[3] for x in inf]
				signalProbs = [x[4] for x in inf]

				model_label = str(key[0] * key[1]) + " particle(s)"
				axs[0].plot(timestepLabels, forwardProbs, label='Forward')
				axs[0].plot(timestepLabels, leftProbs, label='Left')
				axs[0].plot(timestepLabels, rightProbs, label='Right')
				axs[0].plot(timestepLabels, stopProbs, label='Stop')
				axs[0].plot(timestepLabels, signalProbs, label='Signal')

				potential_labels = ["Forward", "Left", "Right", "Stop", "Signal"]
				#goal_label = potential_labels[goal_int]
				fig.set_dpi(100)
				#axs[0].set_title(f"Inferred Ground Truth ({goal_label}) Probability by Time for {model_label}")
				axs[0].set_ylabel("Probability")
				axs[0].set_xlabel("Timestep")
				axs[0].legend()
				general.save_fig(fig, path, tight_layout_kwargs={"rect": [0, 0, 1, 1]})
			# is_nn_accuracy += numberCorrect(actionInferenceCar, car_actions_ints)
			# nn_accuracy += numberCorrect(cross_ent_nn_inferences, car_actions_ints)
			break

	# parse stored data for plotting
	if plotting:
		exit(0)

	print(f"Num signals: {len(nnSignalAccuracy)}")
	print(f"NN Signal Accuracy: {np.mean(nnSignalAccuracy)}; StdDev: {np.std(nnSignalAccuracy)}")
	print(f"Ours (9 particles) Signal Accuracy: {np.mean(oursSignalAccuracy)}; StdDev: {np.std(oursSignalAccuracy)}")
	print(f"-----------------\n")
	randomParticles = {}
	randomKLs = {}
	particles = {}
	pKLs = {}
	print("Variances:")
	for key, accuracies in action_accuracy.items():
		if key == 'is' or key == 'nn':
			continue
		if len(accuracies) == 0:
			continue
		acc = np.mean(accuracies)
		print(f"Key: {key}; Accuracy STD err: {np.std(accuracies) / np.sqrt(len(accuracies))}; KL STD: {np.std(action_kl[key]) / np.sqrt(len(accuracies))}")
		L2_sample, L1_sample = key
		kl = np.mean(action_kl[(L2_sample, L1_sample)])
		if L2_sample == 'random':
			if L1_sample in randomParticles:  # get average of common particles different layers
				randomParticles[L1_sample * 2 * 100 / 72] = (randomParticles[L1_sample] + acc) / 2
				randomKLs[L1_sample* 2 * 100/ 72] = (randomKLs[L1_sample] + kl) / 2
			else:
				randomParticles[L1_sample* 2* 100 / 72] = acc
				randomKLs[L1_sample* 2* 100 / 72] = kl
		else:
			key = (L2_sample * L1_sample* 2)* 100 / 72  # show percent of full IS
			if key in particles:
				particles[key] = (particles[key] + acc) / 2
				pKLs[key] = (pKLs[key] + kl) / 2
			else:
				particles[key] = acc
				pKLs[key] = kl
	print("---------")
	keys = []
	vals = []
	kls = []
	rvals = []
	rkls = []
	for k, v in particles.items():
		keys.append(k)
		vals.append(v)
		kls.append(pKLs[k])
		rvals.append(randomParticles[k])
		rkls.append(randomKLs[k])

	print(f"Ours acc vals: {vals}; Our kls: {kls}")
	print(f"Ours w/o nn acc vals: {rvals}; Our kls: {rkls}")


	

	nnVals = [np.mean(action_accuracy["nn"])] * len(keys) 
	isVals = [np.mean(action_accuracy['is'])] * len(keys)
	nnKL = [np.mean(action_kl['nn'])] * len(keys)


	print(f"NN vals: {np.mean(action_accuracy['nn'])}; NN kls: {np.mean(action_kl['nn'])}")
	print(f"IS vals: {np.mean(action_accuracy['is'])}")
	
	print(f'Key: nn; Accuracy STD: {np.std(action_accuracy["nn"])}; KL STD: {np.std(action_kl["nn"])}')
	print(f'Key: ei; Accuracy STD: {np.std(action_accuracy["is"])}')

	print("--------\n")


	klKeys = list(range(45, 55))
	uniformKL = [np.mean(action_kl['uniform'])] * len(klKeys)

	# accuracy and kl divergence plots
	num_cols = 2
	num_rows = 1
	fig, axss = plt.subplots(
		num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), squeeze=False
	)
	axs = axss.flatten()

	ax = axs[0]
	ax.plot(keys, isVals, linestyle='dashdot', color='gold', label='EI (72 particles)') 
	ax.plot(keys, nnVals, label='NN', color = 'red', linestyle='dashed') 
	ax.scatter(keys, vals, color='blue')
	ax.plot(keys, vals, label='Our Approach w/ NN', color='blue')
	ax.scatter(keys, rvals, color='green')
	ax.plot(keys, rvals, label='Our Approach w/o NN', color='green')
	ax.set_ylabel("Accuracy")
	ax.set_xlabel("% of hypothesis space evaluated")
	ax.legend()

	ax = axs[1]
	ax.plot(keys, nnKL, label='NN', color = 'red', linestyle='dashed') 
	ax.scatter(keys, kls, color='blue')
	ax.plot(keys, kls, label='Our Approach w/ NN', color='blue')
	ax.scatter(keys, rkls, color='green')
	ax.plot(keys, rkls, label='Our Approach w/o NN', color='green')
	ax.plot(klKeys, uniformKL, color='purple', label='Uniform', linestyle='dashed')
	ax.set_ylabel("KL Divergence")
	ax.set_xlabel("% of hypothesis space evaluated")
	ax.legend()


	general.save_fig(fig, f'save/examples/{num_cars}_accuracyHypPct{saveExtension}_clipped_18.png', tight_layout_kwargs={"rect": [0, 0, 1, 1]})




	num_cols = 1
	num_rows = 1
	fig, axss = plt.subplots(
		num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), squeeze=False
	)
	axs = axss.flatten()
	x_ticks = [f'{x}' for x in range(10, 110, 10)]
	for key, timeDict in avgAccuracyPctTime.items():
		xvals = []
		yvals = []
		yerrs = []
		for x, accs in timeDict.items():
			if len(accs) == 0:
				continue
			xvals.append(x)
			yvals.append(np.mean(accs))
			yerrs.append(np.std(accs) / np.sqrt(len(accs)))
		yvals = np.array(yvals)
		yerrs = np.array(yerrs)


		if key == 'nn':
			label = 'ToMnet'
			axs[0].plot(xvals, yvals, label=label, linestyle='dashed', color='red')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='red')
		elif key == 'is':
			label = 'EI'
			axs[0].plot(xvals, yvals, label=label, linestyle='dashdot', color='gold')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='gold')
		elif key == ("random", 6):
			label = f"Ours w/o NN"
			axs[0].plot(xvals, yvals, label=label, color='green')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='green')
		elif key != (3, 2):
			continue
		else:
			label = f"Ours"
			axs[0].plot(xvals, yvals, label=label, color='blue')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='blue')
		print(f"Key {key} Vals: {yvals} Errs: {yerrs}")
	axs[0].set_ylabel("Accuracy", fontsize=18)
	axs[0].set_xlabel("Episode Progress (%)", fontsize=18)
	axs[0].tick_params(length=0)
	axs[0].tick_params(axis="x")
	axs[0].set_xticks(range(10))
	axs[0].set_xticklabels(x_ticks)
	axs[0].legend(fontsize=12)
	general.save_fig(fig, f'save/examples/{num_cars}_accuracyEpPct{saveExtension}_clipped_18.png', tight_layout_kwargs={"rect": [0, 0, 1, 1]})

