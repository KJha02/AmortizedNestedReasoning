import pdb
import sys
import envs.construction_sample
import agents.construction_agent_L1
import envs.construction
import test_construction_desire_pred
from envs.construction import Action
import agents.construction_agent_L2
import copy
import models
import pickle
import utils
import shutil
from utils import construction_data
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from test_construction_agent_L2 import plot_L2_snapshot
from utils.construction_data import multi_agent_state_to_state_tensor, action_to_action_tensor
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import pygame, sys
import os
from pygame.locals import *
import time
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # initialize world with white background
cuda = torch.cuda.is_available()
if cuda:
	device = "cuda"
else:
	device = "cpu"

# # load in test data
# L1_dataset_dir = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning/data/easySynthetic/"
# num_test_data = len(os.listdir(L1_dataset_dir))
# last = 0
# L1_test_dataset = construction_data.ReasoningAboutL1Dataset(
# 	num_colored_blocks=10,
# 	num_possible_block_pairs=45,
# 	num_rows=20,
# 	num_cols=20,
# 	beta=0.01,
# 	utility_mode="ranking",
# 	num_data=num_test_data,
# 	dataset_dir=L1_dataset_dir,
# 	train=False,
# 	seed=123,
# 	device=device,
# 	last=last,
# 	saved_model_dir=None,
# 	L0_inference_model=None,
# 	num_samples=5,
# 	num_samples_L2=2,
# 	human=False,
# 	synthetic=True
# )
# L1_test_dataset.load()
# print("Loaded L1 test data")

# # use a batch size of 1 so that we can consider every sample
# test_dataloader = DataLoader( 
# 	L1_test_dataset,
# 	batch_size=1,
# 	collate_fn=construction_data.multi_collate_last if last else construction_data.multi_collate,
# 	shuffle=False,
# )



def human_interaction(env_multi_agent, agent_0, agent_1):

	string2Action = {'w':envs.construction.Action.UP,
					'a':envs.construction.Action.LEFT,
					's':envs.construction.Action.DOWN,
					'd':envs.construction.Action.RIGHT,
					'q':envs.construction.Action.PUT_DOWN,
					'e':envs.construction.Action.STOP}

	num_agents = env_multi_agent.num_agents
	groundTruthHurting = agent_1.seek_conflict
	obs = env_multi_agent.reset()
	done, cumulative_reward, timestep = False, [0, 0], 0
	num_correct_guesses = 0
	helpingPct = []  # tracking estimation over time
	hurtingPct = []

	rollout = []

	# while not done and timestep < 20 and timestep < min(len(L1_ACTIONS), len(L2_ACTIONS)):
	while not done and timestep < 40:
		state_copy = pickle.loads(pickle.dumps(env_multi_agent.state))
		screen = state_copy.plot()
		pygame.display.update()
		# Make observation for agent 0
		state_0 = obs['agent_0_observation']
		prev_action_0 = obs["prev_actions"][0]
		state_1 = obs['agent_1_observation']
		prev_action_1 = obs["prev_actions"][1]
		# Step the environments for the agents
		agent_0.other_agent.curr_state_L0 = state_0
		agent_0.other_agent.curr_state_L1 = state_1
		agent_1.curr_state_L0 = state_0
		agent_1.curr_state_L1 = state_1


		action_0, action_0_info = agent_0.get_action(
			my_current_state=state_0, my_previous_action=prev_action_0, return_info=True,
			current_state_L1=state_1, prev_action_L1=prev_action_1
		)

		if action_0_info is not None:
			prob_dict = action_0_info["other_agent_seek_conflict"]
			L2_inference_probs = action_0_info["L2_inference"]
			helpingPct.append(prob_dict[0])
			hurtingPct.append(prob_dict[1])
		else:
			helpingPct.append(0.5)
			hurtingPct.append(0.5)

		# Make observation for agent 1
		if prev_action_0 is None:
			observation_1 = None
		else:
			if prev_action_0 == agents.construction_agent_L2.NOOP:
				observation_1 = None
			else:
				observation_1 = envs.construction.ObservationL1(state_0, prev_action_0)
		toQuit = False
		while True:
			try:
				userAction = input("Please select an action \n(W: UP, A: LEFT, S: DOWN, D: RIGHT, Q: DROP, E: Wait)\n")
				userAction = userAction[0]
				if userAction == "m":
					toQuit = True
					break
				action_1 = string2Action[userAction]
				break
			except:
				pass
		# action_1, action_1_info = agent_1.get_action(observation_1, return_info=True)
		if toQuit:
			exit(0)
		agent_1.L1_actions.append(action_1)

		# Build action
		action = {0: action_0, 1: action_1}
		obs, reward, done, info = env_multi_agent.step(action)
		for agent_id in range(num_agents):
			cumulative_reward[agent_id] += reward[agent_id]

		rollout.append((action_1, state_copy, observation_1, reward, helpingPct[-1], hurtingPct[-1], L2_inference_probs, action_0))

		timestep += 1
	state_copy = pickle.loads(pickle.dumps(env_multi_agent.state))
	screen = state_copy.plot()
	pygame.display.update()

	time.sleep(1)

	return rollout, done


def generate_rollout( seek_conflict, env=None):
	num_samples = 1
	num_samples_L2 = 1
	beta_0 = beta_1 = 0.01

	if env is None:
		env = envs.construction_sample.sample_multi_agent_env()
		env.seek_conflict = seek_conflict

	comb_weight = 0 
	entropy_weight = 0
	prior = 0.5

	print("Beginning a new level")
	env.reset()
	grid = env.initial_state.gridworld
	initial_state_L0 = envs.construction.State(
		pickle.loads(pickle.dumps(grid)),
		pickle.loads(pickle.dumps(env.initial_state.agent_locations[0])),
		pickle.loads(pickle.dumps(env.initial_state.colored_blocks)),
	)


	# - Create transitions
	transition_L0 = envs.construction.ConstructionEnv(
		pickle.loads(pickle.dumps(initial_state_L0)), env.colored_block_utilities[0]
	).transition
	transition_L1 = envs.construction.ConstructionEnvL1(
		seek_conflict,
		env.colored_block_utilities[1],
		pickle.loads(pickle.dumps(initial_state_L0)),
		env.colored_block_utilities[0],
		agent_location_L1=pickle.loads(pickle.dumps(env.initial_state.agent_locations[1])),
		agent_inv_L1=pickle.loads(pickle.dumps(env.initial_state.agent_inv[1]))
	).transition

	# if inference_model is not None:
	# 	inference_model = inference_model
	# elif saved_model_dir is not None and inference_model is None:
	# 	inference_model, optimizer, stats, args = test_construction_desire_pred.load_checkpoint(saved_model_dir, device)
	# else:
	inference_model = None
	L1_utility = None

	agent_1 = agents.construction_agent_L1.AgentL1(
		seek_conflict,
		L1_utility,
		env.num_possible_block_pairs,
		initial_state_L0,
		pickle.loads(pickle.dumps(env.initial_state.agent_locations[1])),
		transition_L0,
		transition_L1,
		inference_algorithm="IS",   # make L1 the exact inference
		beta_L0=beta_0,
		beta_L1=beta_1,
		num_samples=num_samples,
		model=inference_model,
		ground_truth_colored_block_utilities_L0=env.colored_block_utilities[0]
	)

	agent_0 = agents.construction_agent_L2.AgentL2(
		env.colored_block_utilities[0],
		initial_state_L0,
		pickle.loads(pickle.dumps(env.initial_state.agent_locations[1])),
		transition_L0,
		transition_L1,
		inference_algorithm="IS",  # make the higher level L2 the online IS + NN
		num_samples=num_samples_L2,
		other_agent_beta_L0=beta_0,
		other_agent_beta_L1=beta_1,
		other_agent_num_samples=num_samples,  
		other_agent_model=inference_model,
		other_agent_inference_algorithm="IS",
		beta_L2=beta_0,
		prior=prior,
		comb_weight=comb_weight,
		entropy_weight=entropy_weight
	)

	rollout, done = human_interaction(env, agent_0, agent_1)
	return rollout


def saveRollout(rollout, desire_int, idx, save_dir):
	single_data_point = {}

	try:
		L1_actions_raw = [x[0] for x in rollout]
		L2_actions_raw = [x[7] for x in rollout]
	except:
		pdb.set_trace()
		L1_actions_raw = [x[0] for x in rollout]
		L2_actions_raw = [x[7] for x in rollout]
	states_raw = [x[1] for x in rollout]

	# for state_id in range(1, len(states_raw)):
	# 	prev_state = states_raw[state_id - 1]
	# 	temp_actions = {0: L2_actions_raw[state_id-1], 1: L1_actions_raw[state_id-1]}
	# 	assert env.transition(prev_state, temp_actions) == states_raw[state_id], f"Transition function failed \n {prev_state} \n {temp_actions} \n {states_raw[state_id]}"

	num_timesteps = len(states_raw)

	states = torch.stack(
		[
			multi_agent_state_to_state_tensor(state_raw, 10, device)
			for state_raw in states_raw
		],
		dim=0,
	)

	# Convert actions to tensors
	L1_actions = torch.stack(
		[action_to_action_tensor(action_raw, device) for action_raw in L1_actions_raw], dim=0,
	)
	L2_actions = torch.stack(
		[action_to_action_tensor(action_raw, device) for action_raw in L2_actions_raw], dim=0,
	)
	desire = torch.tensor([desire_int] * num_timesteps, device=device,)

	socialInference = [x[6] for x in rollout]

	#socialInference = [[x[4], x[5]] for x in rollout]

	final_correct = np.argmax(socialInference[-1]) == desire_int
	num_correct_guesses = np.sum(np.argmax(socialInference, axis=-1) == desire_int)

	single_data_point["final_correct"] = torch.tensor(final_correct, device=device)
	single_data_point["num_correct_guesses"] = torch.tensor(num_correct_guesses, device=device)
	single_data_point["states"] = states.to(device)
	single_data_point["L1_actions"] = L1_actions.to(device)
	single_data_point["L2_actions"] = L2_actions.to(device)
	single_data_point["num_block_pairs"] = torch.tensor(45, device=device)
	single_data_point["desire"] = desire
	single_data_point["IS_inferences"] = torch.tensor(socialInference, device=device)

	path_name = f"{save_dir}{idx}.pik"
	with open(path_name, "wb") as f:  # save each datapoint to a separate file
		pickle.dump(single_data_point, f)
		f.close()



def sample_from_synthetic(seek_conflict):
	batch_to_use = random.randint(0, len(test_dataloader))

	for batch_id, batch in enumerate(test_dataloader):
		if batch_id != batch_to_use:
			continue

		states, L1_actions, desires, IS_inference, num_correct_final, num_correct_overall, L2_actions = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
		state_tensor_rollouts = states[0]
		initial_state = construction_data.multi_agent_state_tensor_to_state(state_tensor_rollouts[0], getL1=False)
		seek_conflict = seek_conflict  # is the user going to help or hurt

		# we don't actually care about the block pair utilities since our accuracy metrics are based on social goal
		colored_block_utilities = {0: envs.construction_sample.sample_block_pair_utilities(45), 
			1: envs.construction_sample.sample_block_pair_utilities(45)
		}

		# we have our multi agent environment
		multi_agent_env = envs.construction.ConstructionMultiAgentEnv(initial_state, colored_block_utilities, seek_conflict=seek_conflict)
		return multi_agent_env




if __name__ == "__main__":
	personSampled = int(sys.argv[1])
	try:
		skipFirst = sys.argv[2] == "trial"
	except:
		skipFirst = False
	save_dir = "data/construction/humanExp/"
	

	print("\nLet's try a case where you're trying to prevent the blue agent from reaching its goal (which you don't know)\n")
	
	num_data_hurt = 5
	if skipFirst:
		num_data_hurt += 1
	num_data_help = 5

	idx = personSampled * 10
	for i in range(num_data_hurt):
		seek_conflict = True
		desire_int = int(seek_conflict)
		# synthetic_env = sample_from_synthetic(seek_conflict)
		rollout = generate_rollout(seek_conflict, env=None)
		if (i > 0 and skipFirst) or (not skipFirst):
			saveRollout(rollout, desire_int, idx, save_dir)  # desire int = 
			idx += 1



	print("\nNow we're going to try a case where you're helping the blue agent complete its goal (which you don't know)\n")

	time.sleep(5)
	for i in range(num_data_help):
		seek_conflict = False
		desire_int = int(seek_conflict)
		# synthetic_env = sample_from_synthetic(seek_conflict)
		rollout = generate_rollout(seek_conflict, env=None)
		saveRollout(rollout, desire_int, idx, save_dir)
		idx += 1


