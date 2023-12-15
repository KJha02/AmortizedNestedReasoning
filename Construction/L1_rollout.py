import pdb

import envs.construction_sample
import agents.construction_agent_L1
import envs.construction
import test_construction_desire_pred
#import train_construction_desire_pred
from envs.construction import Action
import agents.construction_agent_L2
import copy
import models
import pickle
import utils
import shutil
import torch
import random
from test_construction_agent_L2 import plot_L2_snapshot
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import gc

cuda = torch.cuda.is_available()
if cuda:
	torch.cuda.manual_seed(123)  # default seed
	device = "cuda"
else:
	device = "cpu"
# torch.multiprocessing.set_start_method('spawn')


def L1_block_pair_utilities_to_desire_int(colored_block_utilities, num_possible_block_pairs=45):
	# Make a tuple of utility ints with a fixed order based on ALL_BLOCK_PAIRS
	utilities = []
	for block_pair in envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs]:
		utilities.append(colored_block_utilities[block_pair])
	# utilities = tuple(utilities)

	# # Compute desire int based on utilities permutations
	# utilities_permutations = list(
	#     itertools.permutations(envs.construction.ALL_UTILITIES[:num_possible_block_pairs])
	# )
	# final_util_perm = []
	# for util in utilities_permutations:
	#     if util not in final_util_perm:
	#         final_util_perm.append(util)
	# return final_util_perm.index(utilities)
	try:
		res = utilities.index(100) + 2
	except:
		pdb.set_trace()
	return res


def default_L1_rollout(env=None, beta_0=0.01, beta_1=0.01, device="cpu", saved_model_dir=None, inference_model=None, num_samples=5, num_samples_L2=2, gif_number=1, useBFS=False):
	rollouts = []

	desire_ints = []

	if env is None:
		env = envs.construction_sample.sample_multi_agent_env()
	for seek_conflict in [False, True]:
		env.seek_conflict = seek_conflict
		# env = envs.construction_sample.default_multi_agent_env(seek_conflict)
		# heatmap_data_conflict = []
		# heatmap_data_speed = []
		conflict_prior = [0.5]  # uniform between helping and hurting
		comb_weights = [0]  # don't try influencing L1's actions
		entropy_weights = [0]
		for prior in conflict_prior:
			# pct_correct_prior = []
			# num_timestepss = []
			for comb_weight in comb_weights:
				for entropy_weight in entropy_weights:
					print("Making example")
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

					if inference_model is not None:
						inference_model = inference_model
					elif saved_model_dir is not None and inference_model is None:
						inference_model, optimizer, stats, args = test_construction_desire_pred.load_checkpoint(saved_model_dir, device)
					else:
						inference_model = None
					# utils.network.load_model(inference_model, saved_model_dir, device)
					# inference_model = None
					give_L1_util = random.random() < 0  # for now just do Helping or Hurting

					if give_L1_util:
						L1_utility = env.colored_block_utilities[1]
						L1_utility = envs.construction_sample.sample_block_pair_utilities(
						        len(L1_utility), return_prob=False
						    )
						desire_ints.append(L1_block_pair_utilities_to_desire_int(L1_utility, len(L1_utility)))
					else:
						L1_utility = None
						desire_ints.append(int(seek_conflict))

					agent_1 = agents.construction_agent_L1.AgentL1(
												seek_conflict,
												L1_utility,
												env.num_possible_block_pairs,
												initial_state_L0,
												pickle.loads(pickle.dumps(env.initial_state.agent_locations[1])),
												transition_L0,
												transition_L1,
												inference_algorithm="Online_IS+NN",   # make L1 the exact inference
												beta_L0=beta_0,
												beta_L1=beta_1,
												num_samples=num_samples,
												model=inference_model,
												ground_truth_colored_block_utilities_L0=env.colored_block_utilities[0],
												useBFS=useBFS
											)
					# Create agent 0
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
						other_agent_inference_algorithm="Online_IS+NN",
						beta_L2=beta_0,
						prior=prior,
						comb_weight=comb_weight,
						entropy_weight=entropy_weight,
						useBFS=useBFS
					)
					L1_rollout, done = evaluate_fixed_L1(env, agent_0, agent_1, gif_path=f"save/examples/helping{gif_number}.gif", history_path=None)
					rollouts.append((L1_rollout, done))

	return rollouts, env, desire_ints


def evaluate_fixed_L1(env_multi_agent, agent_0, agent_1, gif_path, history_path):
	num_agents = env_multi_agent.num_agents
	groundTruthHurting = agent_1.seek_conflict
	# img_paths = []
	# img_gui_paths = []
	# tmp_dir = utils.general.get_tmp_dir()
	# # actions for helping behavior
	# L1_Helping_Actions = [Action.STOP, Action.UP, Action.UP, Action.STOP, Action.RIGHT, Action.RIGHT, Action.LEFT, Action.LEFT,
	#                       Action.LEFT, Action.LEFT, Action.LEFT, Action.RIGHT, Action.UP,  Action.UP, Action.RIGHT, Action.DOWN, Action.RIGHT]

	# L1_Helping_Short = [Action.STOP, Action.LEFT, Action.LEFT, Action.UP, Action.UP, Action.LEFT, Action.UP, Action.RIGHT, Action.UP]

	# L1_Hurting_Actions = [Action.STOP, Action.UP, Action.UP, Action.UP, Action.LEFT, Action.UP, Action.LEFT, Action.DOWN, Action.LEFT,
	#                       Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.UP, Action.RIGHT,
	#                       Action.UP, Action.STOP]

	# L2_Helping_Actions = [Action.DOWN, Action.DOWN, Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.LEFT, Action.LEFT,
	#                       Action.STOP, Action.DOWN, Action.DOWN, Action.UP, Action.LEFT, Action.LEFT, Action.LEFT, Action.UP, Action.LEFT]

	# L2_Helping_Short = [Action.LEFT, Action.DOWN, Action.DOWN, Action.DOWN, Action.RIGHT, Action.LEFT, Action.DOWN, Action.LEFT, Action.DOWN]

	# L2_Hurting_Actions = [Action.LEFT, Action.DOWN, Action.DOWN, Action.DOWN, Action.RIGHT, Action.LEFT, Action.DOWN, Action.LEFT,
	#               Action.LEFT, Action.DOWN, Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.RIGHT, Action.RIGHT,
	#               Action.UP, Action.UP]

	# if groundTruthHurting:
	#     L1_ACTIONS = L1_Hurting_Actions
	#     L2_ACTIONS = L2_Hurting_Actions
	# else:
	#     # L1_ACTIONS = L1_Helping_Short
	#     # L2_ACTIONS = L2_Helping_Short
	#     # uncomment next two lines for a longer action sequence
	#     L1_ACTIONS = L1_Helping_Actions
	#     L2_ACTIONS = L2_Helping_Actions


	obs = env_multi_agent.reset()
	done, cumulative_reward, timestep = False, [0, 0], 0
	num_correct_guesses = 0
	helpingPct = []  # tracking estimation over time
	hurtingPct = []


	rollout = []

	# while not done and timestep < 20 and timestep < min(len(L1_ACTIONS), len(L2_ACTIONS)):
	while not done and timestep < 3:
		state_copy = pickle.loads(pickle.dumps(env_multi_agent.state))
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

		# if timestep == 7:
		#     pdb.set_trace()
		action_0, action_0_info = agent_0.get_action(
			my_current_state=state_0, my_previous_action=prev_action_0, return_info=True,
			current_state_L1=state_1, prev_action_L1=prev_action_1
		)
		# action_0 = L2_ACTIONS[timestep]

		if action_0_info is not None:
			prob_dict = action_0_info["other_agent_seek_conflict"]
			L2_inference_probs = action_0_info["L2_inference"]
			helpingPct.append(prob_dict[0])
			hurtingPct.append(prob_dict[1])
			# if False in prob_dict:
			# 	helpingPct.append(prob_dict[False])
			# else:
			# 	helpingPct.append(1 - prob_dict[True])
			# if True in prob_dict:
			# 	hurtingPct.append(prob_dict[True])
			# else:
			# 	hurtingPct.append(1 - prob_dict[False])
		else:
			helpingPct.append(0.5)
			hurtingPct.append(0.5)


		# if action_0_info["L2_guess_conflict"] == env_multi_agent.seek_conflict:
		#     num_correct_guesses += 1
		# Make observation for agent 1
		if prev_action_0 is None:
			observation_1 = None
		else:
			if prev_action_0 == agents.construction_agent_L2.NOOP:
				observation_1 = None
			else:
				observation_1 = envs.construction.ObservationL1(state_0, prev_action_0)
		action_1, action_1_info = agent_1.get_action(observation_1, return_info=True)
		# print(agent_1.observations)
		# action_1 = L1_ACTIONS[timestep]
		agent_1.L1_actions.append(action_1)

		# Build action
		action = {0: action_0, 1: action_1}
		obs, reward, done, info = env_multi_agent.step(action)
		for agent_id in range(num_agents):
			cumulative_reward[agent_id] += reward[agent_id]

		# for attempts in range(5):  # 5 attempts to get transition to work
		# 	try:
		# 		assert obs == env_multi_agent.transition(state_copy, action)
		# 		break
		# 	except:
		# 		if attempts < 4:
		# 			pass
		# print(obs)

		rollout.append((action_1, state_copy, observation_1, reward, helpingPct[-1], hurtingPct[-1], L2_inference_probs, action_0))

		# img_path = f"{tmp_dir}/{timestep}.png"
		# img_path_gui = f"{tmp_dir}/{timestep}_gui.png"
		# plot_L2_snapshot(
		#     img_path,
		#     img_path_gui,
		#     state_copy,
		#     env_multi_agent.colored_block_utilities[0],
		#     action_1,
		#     action_1_info,
		#     action_0,
		#     action_0_info
		# )
		# img_paths.append(img_path)
		# img_gui_paths.append(img_path_gui)
		timestep += 1

		
		# print(
		#     f"t = {timestep} | action = {action} | reward = {reward} | "
		#     f"cumulative_reward = {cumulative_reward} | done = {done}"
		# )
		# print(
		#     f"inventory = {env_multi_agent.state.agent_inv} | block_locations = {env_multi_agent.state.colored_blocks}"
		# )


	# if groundTruthHurting == True:  # plot hurting probabilities over time if GT social goal is hurting
	#     xvals = [x+1 for x in range(len(hurtingPct))]
	#     plt.plot(xvals, hurtingPct)
	#     plt.title("L2's belief about L1 hurting over time \n Ground Truth for L1 is Hurting")
	#     plt.savefig(history_path)
	# else:
	#     xvals = [x+1 for x in range(len(helpingPct))]
	#     plt.plot(xvals, helpingPct)
	#     plt.title("L2's belief about L1 helping over time \n Ground Truth for L1 is Helping")
	#     plt.savefig(history_path)

	# Make gif
	# utils.general.make_gif(img_paths, gif_path, 3)
	# shutil.rmtree(tmp_dir)

	# # Return whether agent 0 deceived agent 1, return average correct guesses of seek_conflict
	# return cumulative_reward[0], num_correct_guesses / timestep, timestep
	return rollout, done


if __name__ == "__main__":
	saved_model_dir = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning/save/construction/30.0kDat_smallerModel_128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/checkpoints/best_acc.pik"

	saved = []

	for i in range(6):
		new_rolls, env, desires = default_L1_rollout(saved_model_dir=saved_model_dir, gif_number=i)
	
	# print(desires)
	# exit(0)

	# for i in range(2):
	# 	colored_block_utilities_0 = envs.construction_sample.sample_block_pair_utilities(
	# 	45, return_prob=False
	# 	)
	# 	colored_block_utilities_1 = colored_block_utilities_0
	# 	colored_block_utilities = {
	# 		0: colored_block_utilities_0,
	# 		1: colored_block_utilities_1,
	# 	}
	# 	env.colored_block_utilities = colored_block_utilities
	# 	new_rolls, env, desire_ints = default_L1_rollout(env=env, saved_model_dir=saved_model_dir)
	# 	for r in new_rolls:
	# 		rollout, done = r
	# 		print(done)

