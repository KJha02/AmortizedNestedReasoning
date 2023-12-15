from scenario import Scenario1, Action, get_partial_states
import random
import numpy as np
import _pickle as pickle
from drivingAgents.car_agent_L0 import AgentL0
import reasoning_about_car_L0
import pdb
import time
import torch
from tqdm import tqdm
import car_utils.general as general
from car_models.ToMnet_car import ToMnet_state_pred, ToMnet_exist_pred
import torch.optim as optim
import shutil
from pathlib import Path



def init_statePred(args, device='cpu'):
	state_model = ToMnet_state_pred(hidden_dim=args.hidden_dim)
	state_model.to(device)
	state_optimizer = optim.Adam(state_model.parameters(), lr=args.lr)
	return state_model, state_optimizer

def init_existPred(args, device='cpu'):
	exist_model = ToMnet_exist_pred(hidden_dim=args.hidden_dim)
	exist_model.to(device)
	exist_optimizer = optim.Adam(exist_model.parameters(), lr=args.lr)
	return exist_model, exist_optimizer

def load_belief_checkpoint(path, device, num_tries=3, exist_model=False):
	for i in range(num_tries):
		try:
			checkpoint = torch.load(path, map_location=device)
			break
		except Exception as e:
			print(f"Error {e}")
			wait_time = 2 ** i
			print(f"Waiting for {wait_time} seconds")
			time.sleep(wait_time)
	args = checkpoint["args"]
	if exist_model:
		model, optimizer = init_existPred(args, device)
	else:
		model, optimizer = init_statePred(args, device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	stats = checkpoint["stats"]
	return model, optimizer, stats, args


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

	# totCorrect = 0 
	# for i in range(len(x)):
	# 	totCorrect += np.argmax(x[i]) == y[i]
	# return totCorrect / len(x)


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


def generate_scenario1_rollout(car_exist_prior=0.5, car_state_model=None, car_exist_model=None, sampled_actions=10, lookAheadDepth=5, gif_dir=None):
	# Init the scenario
	scenario = Scenario1()
	scenario.reset()

	idToCar = {}
	idToAgent = {}

	for car in scenario.w.dynamic_agents:
		agent = AgentL0(car.ID, car.actionGoal.value, scenario.transition, scenario.initialLocMapping, 
		carExistPrior=car_exist_prior, sampled_actions=sampled_actions, lookAheadDepth=lookAheadDepth,
		state_model=car_state_model, exist_model=car_exist_model)
		idToCar[car.ID] = car 
		idToAgent[car.ID] = agent

	states = None
	full_observed_state_history = []  # third person perspective
	full_action_history = []
	action_history = {}
	belief_tensor_history = {}
	ground_truth_goals = {}  # what do both agents actually want to do
	prev_actions = {}
	test_id = None
	target_id = None
	for cID in idToAgent:
		if test_id is None:
			test_id = cID
		elif target_id is None:
			target_id = cID
		action_history[cID] = [Action.FORWARD]
		belief_tensor_history[cID] = []
		ground_truth_goals[cID] = idToAgent[cID].goalActionString
		prev_actions[cID] = Action.FORWARD


	done = False
	for step_number in tqdm(range(30)):
		full_observed_state_history.append(scenario.w.clone())
		action_dict = {}
		# if states is None:
		# 	for cID in idToAgent:  # all agents move forward initially
		# 		action_dict[cID] = Action.FORWARD
		# else:
		for cID in idToAgent:
			try:
				c_state = states[cID]
				c_observation = c_state[2][1] # 3rd element of state is partial observation - (agentID, obs)
			except:
				c_observation = get_partial_states(scenario.w, id=cID)
				
			if cID == test_id:
				partial_L1_state = c_observation


			for other_cars in c_observation.dynamic_agents: # we do this so our agent imagines other agents keep maintaining their actions
				if other_cars.ID in prev_actions:
					other_cars.prev_action = prev_actions[other_cars.ID]
			
			c_action, c_info = idToAgent[cID].get_action(observation=c_observation, return_info=True, prev_actions=prev_actions)
			action_dict[cID] = c_action
			action_history[cID].append(c_action)
			belief_tensor_history[cID].append(c_info["belief_tensor"])  # store so we can amortize beliefs

		next_states, reward, done, info = scenario.step(action_dict)
		states = next_states
		for k in prev_actions.keys():
			try:
				assert k in action_dict
			except:
				pdb.set_trace()

		prev_actions = pickle.loads(pickle.dumps(action_dict))
		full_action_history.append(prev_actions)

		if done:
			break
	return scenario, full_observed_state_history, action_history, ground_truth_goals, done, belief_tensor_history, full_action_history


def make_car_gif(gif_path, save_dir, states_raw):
	tmp_dir = general.get_tmp_dir()
	img_paths  = []
	for timestep, state in enumerate(states_raw):
		state.render(save_dir + tmp_dir, timestep)
		img_paths.append(f"{save_dir}{tmp_dir}/{timestep}_gui.png")

	general.make_gif(img_paths, save_dir + gif_path, 3)
	# shutil.rmtree(save_dir + tmp_dir)
	

if __name__ == "__main__":
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)
	cuda = torch.cuda.is_available()
	if cuda:
		torch.cuda.manual_seed(1)
		device = "cuda"
	else:
		device = "cpu"

	num_cars = 3

	num_inference_samples = 3
	
	car_exist_prior = 0.6

	state_belief_save_dir = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning/CARLO/save/stateEstimation/3Car_StateBelief_3.0kDat_64dim_0.1lr_128bSize/num_sampled_actions=1,lookAheadDepth=10,beta=0.01/checkpoints/"
	state_model_path = state_belief_save_dir + "best_acc_state.pik"
	exist_model_path = state_belief_save_dir + "best_acc_exist.pik"
	state_model, _, _, _ = load_belief_checkpoint(state_model_path, device, exist_model=False)
	exist_model, _, _, _ = load_belief_checkpoint(exist_model_path, device, exist_model=True)

	sampled_actions = 20
	lookAheadDepth = 3


	num_trials = 3

	goal_accuracy = {}
	action_accuracy = {}
	for n in range(num_cars):
		goal_accuracy[n] = 0.0 
		action_accuracy[n] = 0.0 


	car1_IS_accuracy = 0.0
	car1_action_accuracy = 0.0 
	car2_IS_accuracy = 0.0
	car2_action_accuracy = 0.0 

	for trial in tqdm(range(num_trials)):
		rollout = generate_scenario1_rollout(car_exist_prior, state_model, exist_model, sampled_actions, lookAheadDepth)
		scenario, full_observed_state_history, action_history, ground_truth_goals, done, belief_tensor_history, full_action_history = rollout

		ids = list(action_history.keys())

		goalInferenceCar1, goalInferenceCar2, goalInferenceCar3 = None, None, None
		actionInferenceCar1, actionInferenceCar2, actionInferenceCar3 = None, None, None
		car1_goal_idx, car2_goal_idx, car3_goal_idx = None, None, None

		partial_obs_history = None
		for car_number, id in enumerate(ids):
			if partial_obs_history is None:  # inferring from first car perspective
				partial_obs_history = [get_partial_states(s, id=id) for s in full_observed_state_history]
			

			car_actions = action_history[id]
			car_goal = ground_truth_goals[id]
			car_goal_idx = reasoning_about_car_L0.GOAL_SPACE.index(car_goal)
			action_vals = list(Action)
			car_actions_ints = np.array([action_vals.index(Action(a.value)) for a in car_actions])


			# goalInferenceCar = [[1/3, 1/3, 1/3]]* len(full_observed_state_history)
			# actionInferenceCar = [[1/4, 1/4, 1/4, 1/4, 0.0]] * len(full_observed_state_history)

			goalInferenceCar, actionInferenceCar = reasoning_about_car_L0.get_car_L0_is_inference(
				scenario, 
				partial_obs_history, 
				car_actions, 
				id, 
				num_samples=num_inference_samples, 
				sampled_actions=sampled_actions, 
				lookAheadDepth=lookAheadDepth,
				full_action_history=full_action_history,
				state_belief_model=state_model,
				exist_belief_model=exist_model
			)
 
			if goalInferenceCar1 is None:  # car 3
				goalInferenceCar1 = goalInferenceCar
				actionInferenceCar1 = actionInferenceCar
				car1_goal_idx = car_goal_idx
			elif goalInferenceCar2 is None:  # car 2
				goalInferenceCar2 = goalInferenceCar
				actionInferenceCar2 = actionInferenceCar
				car2_goal_idx = car_goal_idx
			else: # car 3
				goalInferenceCar3 = goalInferenceCar
				actionInferenceCar3 = actionInferenceCar
				car3_goal_idx = car_goal_idx

			goal_accuracy[car_number] += percentCorrect(goalInferenceCar, [car_goal_idx] * len(goalInferenceCar))
			action_accuracy[car_number] += percentCorrect(actionInferenceCar[:-1], car_actions_ints[1:])

		gif_path = "base_car.gif"
		save_dir = "save/scenario1/"
		# general.make_scenario1_gif(
		# 	gif_path, 
		# 	save_dir, 
		# 	states_raw=full_observed_state_history, 
		# 	agent1_goal_distribs=goalInferenceCar1,
		# 	agent1_action_distribs=actionInferenceCar1,
		# 	agent1_goal_int=car1_goal_idx,
		# 	agent2_goal_distribs=goalInferenceCar2,
		# 	agent2_action_distribs=actionInferenceCar2,
		# 	agent2_goal_int=car2_goal_idx,
		# 	agent3_goal_distribs=goalInferenceCar3,
		# 	agent3_action_distribs=actionInferenceCar3,
		# 	agent3_goal_int=car3_goal_idx,
		# )
		# car1_actions = action_history[ids[0]]
		# car1_goal = ground_truth_goals[ids[0]]
		# car1_goal_idx = reasoning_about_car_L0.GOAL_SPACE.index(car1_goal)

		# action_vals = list(Action)
		# car1_actions_ints = np.array([action_vals.index(Action(a.value)) for a in car1_actions])

		# goalInferenceCar1, actionInferenceCar1 = reasoning_about_car_L0.get_car_L0_is_inference(
		# 	scenario, 
		# 	full_observed_state_history, 
		# 	car1_actions, 
		# 	ids[0], 
		# 	num_samples=num_inference_samples, 
		# 	sampled_actions=sampled_actions, 
		# 	lookAheadDepth=lookAheadDepth,
		# )
		# car1_IS_accuracy += percentCorrect(goalInferenceCar1, car1_goal_idx)

		# car1_action_accuracy += percentCorrect(actionInferenceCar1[:-1], car1_actions_ints[1:])

		# car2_actions = action_history[ids[1]]
		# car2_goal = ground_truth_goals[ids[1]]
		# car2_goal_idx = reasoning_about_car_L0.GOAL_SPACE.index(car2_goal)

		# car2_actions_ints = np.array([action_vals.index(Action(a.value)) for a in car2_actions])

		# goalInferenceCar2, actionInferenceCar2 = reasoning_about_car_L0.get_car_L0_is_inference(
		# 	scenario, 
		# 	full_observed_state_history, 
		# 	car2_actions, 
		# 	ids[1], 
		# 	num_samples=num_inference_samples, 
		# 	sampled_actions=sampled_actions, 
		# 	lookAheadDepth=lookAheadDepth,
		# )
		# car2_IS_accuracy += percentCorrect(goalInferenceCar2, car2_goal_idx)
		# car2_action_accuracy += percentCorrect(actionInferenceCar2[:-1], car2_actions_ints[1:])


	for car_number in goal_accuracy:
		avg_goal_acc = goal_accuracy[car_number] / num_trials
		avg_action_acc = action_accuracy[car_number] / num_trials
		print(f"\n Accuracy for IS Goal Inference of Car {car_number} from 3rd person full observation: {avg_goal_acc}")
		print(f"\n Accuracy for IS Action Inference of Car {car_number} from 3rd person full observation: {avg_action_acc}")


	# car1_IS_accuracy /= num_trials
	# car1_action_accuracy /= num_trials
	# car2_IS_accuracy /= num_trials
	# car2_action_accuracy /= num_trials

	# print(f"\n Accuracy for IS Goal Inference of Car 1 from 3rd person full observation: {car1_IS_accuracy}")
	# print(f"\n Accuracy for IS Goal Inference of Car 2 from 3rd person full observation: {car2_IS_accuracy}")

	# print(f"\n Accuracy for IS Action Inference of Car 1 from 3rd person full observation: {car1_action_accuracy}")
	# print(f"\n Accuracy for IS Action Inference of Car 2 from 3rd person full observation: {car2_action_accuracy}")

		

