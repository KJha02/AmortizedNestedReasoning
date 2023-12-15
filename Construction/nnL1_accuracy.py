import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import tqdm
from pathlib import Path
import _pickle as pickle
import envs.construction_sample
from utils.construction_data import block_pair_utilities_to_desire_int, get_rollout_gt_inference, multi_collate, multi_collate_last
import pdb
import models
import test_reasoning_about_construction_L0
import test_reasoning_about_construction_L1
from test_reasoning_about_construction_L0 import particle_inference, online_importance_sampling
from test_reasoning_about_construction_L1 import L1_particle_inference, get_gt_L1_is_inference, L1_online_importance_sampling
from test_reasoning_about_construction_L1 import L1_inference_to_posterior_distrib
import utils
from utils import construction_data
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import time
import shutil
import sys
from scipy.stats import entropy
import train_construction_desire_pred
from train_construction_desire_pred import get_gt_inference
from test_construction_agent_L2 import plot_L2_snapshot
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

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


# setting seed

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
cuda = torch.cuda.is_available()
if cuda:
	torch.cuda.manual_seed(123)
	device = "cuda"
else:
	device = "cpu"


# basic variables
num_colored_block_locations = 10
num_possible_block_pairs = 45
num_train_data = 175
num_test_data = int(0.2 * num_train_data)
last = 0

BIG_STORAGE_DIR = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning"


# load in models for inference
L0_model_dir = "/save/construction/30.0kDat_smallerModel_128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/"
model_dir_list = [L0_model_dir]
for pct in [2,5,10]:
	model_dir_list.append(f"/save/construction/60.0kDat_L0_KLLess_{pct}_pct_128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/")
L0_model_dir = model_dir_list[3]

L0_model_path = BIG_STORAGE_DIR + L0_model_dir + "checkpoints/best_acc.pik"
L0_inference_model, L0_optimizer, L0_stats, L0_args = train_construction_desire_pred.load_checkpoint(L0_model_path, device)
L0_inference_model.eval()

L1_model_dir = "/save/construction/1.02kDat_L1dataGen5L22_128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/"

# model_dir_list = [L1_model_dir]
# for pct in [2, 5, 10]:
# 	model_dir_list.append(f"/save/construction/1.02kDat_KLLess_{pct}_pct_128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/")
# L1_model_dir = model_dir_list[0]
# max_acc = []

L1_model_path = BIG_STORAGE_DIR + L1_model_dir + "checkpoints/best_acc.pik"
L1_inference_model, L1_optimizer, L1_stats, L1_args = train_construction_desire_pred.load_checkpoint(L1_model_path, device, L1=True)
L1_inference_model.eval()


L1_nll_model_dir = "/save/construction/1.02kDat_nllLoss128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/"
L1_nll_model_path = BIG_STORAGE_DIR + L1_nll_model_dir + "checkpoints/best_acc.pik"
cross_ent_model, cross_ent_optimizer, cross_ent_stats, cross_ent_args = train_construction_desire_pred.load_checkpoint(L1_nll_model_path, device, L1=True)
cross_ent_model.eval()


# load in test data
L1_dataset_dir = BIG_STORAGE_DIR + "/data/construction"

local_L1_dataset_dir = "data/construction"


L1_test_dataset = construction_data.ReasoningAboutL1Dataset(
	num_colored_blocks=num_colored_block_locations,
	num_possible_block_pairs=num_possible_block_pairs,
	num_rows=20,
	num_cols=20,
	beta=0.01,
	utility_mode="ranking",
	num_data=num_test_data,
	dataset_dir=L1_dataset_dir,
	train=False,
	seed=123,
	device=device,
	last=last,
	saved_model_dir=L0_model_path,
	L0_inference_model=L0_inference_model,
	num_samples=5,
	num_samples_L2=2,
	human=False,
	synthetic=False
)
L1_test_dataset.load()
print("Loaded L1 test data")

# use a batch size of 1 so that we can consider every sample
test_dataloader = DataLoader( 
	L1_test_dataset,
	batch_size=1,
	collate_fn=multi_collate_last if last else multi_collate,
	shuffle=True,
)
method_Accuracy_Runtime = {}  # dictionary mapping method (i.e. "IS" "NN", "IS + NN 2 Particles", etc.) to tuple of (accuracy as decimal, avg runtime (s))

# iterate through test dataset and generate accuracies
nn_runtime = 0.0  # just NN
full_is_runtime = 0.0  # IS 2 particles for L2, 45 particles for L1
is_nn_2_random_runtime = 0.0  # IS + NN 2 particles for L2, Random L1 inference


nn_accuracy = 0
cross_accuracy = 0
nn_kl_div = 0.0
full_is_accuracy = 0
is_nn_2_random_accuracy = 0
is_nn_2_random_kl = 0


num_predictions = 0

num_batches = 0


runtimeDict = {}
accuracyDict = {}
accuracyDictList = {}
klDict = {}
RandomruntimeDict = {}
RandomaccuracyDict = {}
RandomaccuracyDictList = {}
RandomklDict = {}
avgAccuracyPctTime = {}
avgAccuracyPctTime['is'] = {i: [] for i in range(10)}
avgAccuracyPctTime['nn'] = {i: [] for i in range(10)}
for L2_sample in range(2, 3):
	for L1_sample in [1, 2, 3, 4, 5, 15, 30]:
	# for L1_sample in [5, 10, 15, 30]:
	# for L1_sample in [5]:
		runtimeDict[(L2_sample, L1_sample)] = 0.0
		accuracyDict[(L2_sample, L1_sample)] = 0.0
		accuracyDictList[(L2_sample, L1_sample)] = []
		klDict[(L2_sample, L1_sample)] = 0.0
		RandomruntimeDict[(L2_sample, L1_sample)] = 0.0
		RandomaccuracyDict[(L2_sample, L1_sample)] = 0.0
		RandomaccuracyDictList[(L2_sample, L1_sample)] = []
		RandomklDict[(L2_sample, L1_sample)] = 0.0
		avgAccuracyPctTime[(L2_sample, L1_sample)] = {i: [] for i in range(10)}


action_kl = {'nn': []}
action_kl['uniform'] = []   # uniform probs
action_accuracy = {'is':[]}
action_accuracy['nn'] = []
for n in range(2, 3):
	for m in [1, 2, 3, 4, 5, 15, 30]:
	# for m in [5]:
		# if (n, m) != (3,3):
		# 	continue
		action_accuracy[(n,m)] = []
		action_kl[(n,m)] = []
		avgAccuracyPctTime[(n,m)] = {i: [] for i in range(10)}
		action_accuracy[('random', n*m)] = []
		action_kl[('random', n*m)] = []
		avgAccuracyPctTime[('random', n*m)] = {i: [] for i in range(10)}


accuracyDictList["IS"] = []
accuracyDictList["NN"] = []


is_nn_2_random_runtimes = []

def summarizeAccuracy():
	temp_nn_accuracy = nn_accuracy / num_batches
	temp_nn_kl = nn_kl_div / num_batches
	temp_full_is_accuracy = full_is_accuracy/ num_batches
	# temp_is_nn_2_random_accuracy = is_nn_2_random_accuracy / num_batches
	temp_cross_accuracy = cross_accuracy / num_batches

	temp_nn_runtime = nn_runtime / num_batches
	temp_full_is_runtime = full_is_runtime / num_batches
	# temp_is_nn_2_random_runtime = is_nn_2_random_runtime / num_batches


	print(f"Cross Ent accuracy {temp_cross_accuracy}")

	print(f"Full IS Accuracy: {temp_full_is_accuracy}")
	print(f"NN Accuracy: {temp_nn_accuracy} NN Avg KL Div: {temp_nn_kl}")
	

	# print(f'Full IS Curr Ep Accuracy {accuracyDictList["IS"][-1]}')
	# print(f'NN Curr Ep Accuracy {accuracyDictList["NN"][-1]}')



	for L2_sample in range(2, 3):
		for L1_sample in [1, 2, 3, 4, 5, 15, 30]:
		# for L1_sample in [5, 10, 15, 30]:
		# for L1_sample in [5]:
			temp_runtime = runtimeDict[(L2_sample, L1_sample)] / num_batches
			temp_accuracy = accuracyDict[(L2_sample, L1_sample)] / num_batches
			temp_kl_div = klDict[(L2_sample, L1_sample)] / num_batches
			temp_std_error = np.std(accuracyDictList[(L2_sample, L1_sample)])
			print(f"IS + NN {L2_sample}x{L1_sample} Accuracy: {temp_accuracy} std ({temp_std_error}), IS + NN {L2_sample}x{L1_sample} Avg KL Div: {temp_kl_div}")

			# print(f'IS + NN {L2_sample}x{L1_sample} Curr Ep Accuracy {accuracyDictList[(L2_sample, L1_sample)][-1]}')


			# temp_Randomruntime = RandomruntimeDict[(L2_sample, L1_sample)] / num_batches
			# temp_Randomaccuracy = RandomaccuracyDict[(L2_sample, L1_sample)] / num_batches
			# temp_Randomkl_div = RandomklDict[(L2_sample, L1_sample)] / num_batches
			# temp_random_std_error = np.std(RandomaccuracyDictList[(L2_sample, L1_sample)])
			# print(f"IS + NN {L2_sample}x{L1_sample} Random_L1 Accuracy: {temp_Randomaccuracy} std ({temp_random_std_error}), IS + NN {L2_sample}x{L1_sample} Random_L1 Avg Runtime: {temp_Randomruntime}, IS + NN {L2_sample}x{L1_sample} Random_L1 Avg KL Div: {temp_Randomkl_div}")
	print("\n")

if __name__ == "__main__":

	beta_L0 = float(sys.argv[1])
	beta_L1 = float(sys.argv[2])

	print(f"Tuning for L0 beta {beta_L0} L1 beta {beta_L1}")
	total_data = len(test_dataloader) - 40
	for epoch in tqdm.tqdm(range(1)):
		# print(f"\n-----\nEpoch {epoch}")


		timeRollout = False
		sampled_batch = random.sample(range(0, len(test_dataloader)), 100)
		#sampled_batch = [random.randint(0, len(test_dataloader))]
		#print(total_data // 2)


		for batch_id, batch in enumerate(test_dataloader):
			print(f"Batch {batch_id}")
			
			if batch_id == 100:
				break
			# if batch_id not in sampled_batch:
				# continue
			# if batch_id >= 20 and batch_id < 40:
			# 	continue
			# if batch_id < 40:
			# 	continue
			timeRollout = False
			
			rawInf = {}
			randomRawInf = {}


			states, L1_actions, desires, IS_inference, num_correct_final, num_correct_overall, L2_actions = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
			lens = torch.LongTensor([s.shape[0] for s in states]).to('cpu')
			actions_2d = [
				utils.network.expand_batch(a, (20, 20, 6))
				for a in L1_actions
			]

			# method 1 - just NN
			nn_start_time = time.time()
			log_prob = cross_ent_model(states, actions_2d, lens, last=last)  # the final inference prediction for the model
			nn_inferences = torch.softmax(log_prob, 1).cpu().detach().numpy()
			nn_inferences /= nn_inferences.sum(axis=1, keepdims=True)
			rawInf["nn"] = nn_inferences
			nn_runtime += time.time() - nn_start_time  # how long did it take to get nn prediction



			cross_log_prob = cross_ent_model(states, actions_2d, lens, last=last)
			cross_inferences = torch.softmax(cross_log_prob, 1).cpu().detach().numpy()
			cross_inferences /= cross_inferences.sum(axis=1, keepdims=True)

			uniform_probs = np.full_like(cross_inferences, 1/2)


			# method 2 - just IS with full 2 x 45 particles
			state_tensor_rollouts = states[0]
			# states_raw = [construction_data.multi_agent_state_tensor_to_state(s) for s in state_tensor_rollouts]
			L1_action_tensor_rollouts = L1_actions[0]
			L2_action_tensor_rollouts = L2_actions[0]
			L1_actions_raw = [construction_data.action_tensor_to_action(a) for a in L1_action_tensor_rollouts]
			L2_actions_raw = [construction_data.action_tensor_to_action(a) for a in L2_action_tensor_rollouts]

			L1_states_raw = []
			states_raw = []
			for i, s in enumerate(state_tensor_rollouts):
				if i == 0:
					multi_roll = construction_data.multi_agent_state_tensor_to_state(s, getL1=False)
					L1_roll = construction_data.multi_agent_state_tensor_to_state(s, getL1=True)
				else:
					multi_roll = construction_data.multi_agent_state_tensor_to_state(s, getL1=False, L2_actions=L2_actions_raw[i-1])
					L1_roll = construction_data.multi_agent_state_tensor_to_state(s, getL1=True, L2_actions=L2_actions_raw[i-1])
				
				L1_states_raw.append(L1_roll)
				states_raw.append(multi_roll)
			

			rollout_conflict_int = desires[0]
			conflict_int_cpu = rollout_conflict_int.item()
			seek_conflict = bool(conflict_int_cpu)
			initial_multi_state = states_raw[0]
			initial_L0_state= envs.construction.State(initial_multi_state.gridworld, initial_multi_state.agent_locations[0], initial_multi_state.colored_blocks, initial_multi_state.agent_inv[0])

			# we don't actually care about the block pair utilities since our accuracy metrics are based on social goal
			colored_block_utilities = {0: envs.construction_sample.sample_block_pair_utilities(num_possible_block_pairs), 
				1: envs.construction_sample.sample_block_pair_utilities(num_possible_block_pairs)
				}

			rollout_env = envs.construction.ConstructionEnvL1(True, colored_block_utilities[1], initial_L0_state, colored_block_utilities[0], 
				initial_multi_state.agent_locations[1], initial_multi_state.agent_inv[1]) 



			# # NEED TO COMMENT OUT LATER
			# foundEasy = False
			# for s in states_raw:
			# 	# we found a state where an agent picked up a block and is helping
			# 	if s.agent_inv[1] is not None and not seek_conflict:  
			# 		foundEasy = True
			# 		break
			# if foundEasy:
			# 	print("foundEasy")
			# 	copiedDataPath = f"{easyDir}{easyID}.pik"
			# 	single_data_point = {}
			# 	single_data_point["final_correct"] = torch.tensor(num_correct_final)
			# 	single_data_point["num_correct_guesses"] = torch.tensor(num_correct_overall)
			# 	single_data_point["states"] = states[0]
			# 	single_data_point["L1_actions"] = L1_actions[0]
			# 	single_data_point["L2_actions"] = L2_actions[0]
			# 	single_data_point["num_block_pairs"] = torch.tensor(45, device=device)
			# 	single_data_point["desire"] = rollout_conflict_int
			# 	single_data_point["IS_inferences"] = torch.tensor(IS_inference, device=device)
			# 	with open(copiedDataPath, "wb") as f:  # save each datapoint to a separate file
			# 		pickle.dump(single_data_point, f)
			# 		f.close()
			# 	easyID += 1
			# continue






			full_is_start_time= time.time()
			full_IS_inferences = get_gt_L1_is_inference(rollout_env, L1_states_raw, L1_actions_raw, num_samples_L2=2, num_samples_L1=45, beta_L0=beta_L0, beta_L1=beta_L1)
			rawInf["is"] = full_IS_inferences
			full_is_runtime += time.time() - full_is_start_time

			# get accuracy
			temp_nn_accuracy = numberCorrect(nn_inferences, conflict_int_cpu)
			nn_accuracy += temp_nn_accuracy
			cross_accuracy += numberCorrect(cross_inferences, conflict_int_cpu)  # nll loss

			nn_kl_div += avg_KL(full_IS_inferences, nn_inferences)
			temp_full_is_accuracy = numberCorrect(full_IS_inferences, conflict_int_cpu)
			full_is_accuracy += temp_full_is_accuracy
			accuracyDictList["IS"].append(temp_full_is_accuracy)
			accuracyDictList["NN"].append(temp_nn_accuracy)



			temp_accuracy = numberCorrect(cross_inferences, conflict_int_cpu)
			temp_kl = avg_KL(full_IS_inferences, cross_inferences)
			action_kl['nn'].append(temp_kl)
			action_accuracy['nn'].append(temp_accuracy)

			temp_kl = avg_KL(full_IS_inferences, uniform_probs)
			action_kl['uniform'].append(temp_kl)

			temp_accuracy = numberCorrect(full_IS_inferences, conflict_int_cpu)
			action_accuracy["is"].append(temp_accuracy)

			# # method 3 - IS + NN with diferent amounts of particles for L1 reasoning about L0 [5, 15, 30]
			for L2_sample in range(2, 3):
				for L1_sample in [1, 2, 3, 4, 5, 15, 30]:
				# for L1_sample in [5, 10, 15, 30]:
				# for L1_sample in [5]:
					startTime = time.time()
					raw = L1_particle_inference(rollout_env, L1_states_raw, L1_actions_raw, num_samples_L2=L2_sample, num_samples_L1=L1_sample,
						other_agent_inference_algorithm="Online_IS+NN",output_every_timestep= True,
						other_agent_inference_model=L0_inference_model, beta_L0=beta_L0, beta_L1=beta_L1
					)
					temp_runtime = time.time() - startTime
					runtimeDict[(L2_sample, L1_sample)] += temp_runtime

					inference = L1_inference_to_posterior_distrib(raw)

					rawInf[(L2_sample, L1_sample)] = inference

					temp_accuracy = numberCorrect(inference, conflict_int_cpu)
					temp_kl = avg_KL(full_IS_inferences, inference)

					accuracyDict[(L2_sample, L1_sample)] += temp_accuracy
					accuracyDictList[(L2_sample, L1_sample)].append(temp_accuracy)
					klDict[(L2_sample, L1_sample)] += temp_kl


					action_accuracy[(L2_sample,L1_sample)].append(temp_accuracy)
					action_kl[(L2_sample,L1_sample)].append(temp_kl)



					# if L1_sample % 5 == 0:
					# 	print(f"IS + NN {L2_sample}x{L1_sample} Accuracy: {temp_accuracy}, IS + NN {L2_sample}x{L1_sample} Avg Runtime: {temp_runtime}, IS + NN {L2_sample}x{L1_sample} Avg KL Div: {temp_kl}")


					# # startTime = time.time()
					# Randomraw = L1_particle_inference(rollout_env, L1_states_raw, L1_actions_raw, num_samples_L2=L2_sample, num_samples_L1=L1_sample,
					# 	other_agent_inference_algorithm="random",output_every_timestep= True,
					# 	other_agent_inference_model=L0_inference_model, beta_L0=beta_L0, beta_L1=beta_L1
					# )
					# RandomruntimeDict[(L2_sample, L1_sample)] += time.time() - startTime

					# Randominference = L1_inference_to_posterior_distrib(Randomraw)
					# randomRawInf[(L2_sample, L1_sample)] = Randominference
					# rawInf[("random", L2_sample * L1_sample)] = Randominference

					# temp_random_accuracy = numberCorrect(Randominference, conflict_int_cpu)
					# RandomaccuracyDict[(L2_sample, L1_sample)] += temp_random_accuracy
					# RandomaccuracyDictList[(L2_sample, L1_sample)].append(temp_random_accuracy)
					# RandomklDict[(L2_sample, L1_sample)] += avg_KL(full_IS_inferences, Randominference)

					# action_accuracy[("random", L2_sample*L1_sample)].append(temp_random_accuracy)
					# action_kl[("random", L2_sample*L1_sample)].append(avg_KL(full_IS_inferences, Randominference))


			for key, inf in rawInf.items():
				for t in range(1, len(inf) + 1):
					for pct in range(1, 11):
						if (t <= pct * len(inf) / 10):
							avgAccuracyPctTime[key][pct-1].append(numberCorrect(inf[t-1:t], conflict_int_cpu))
							break
			if batch_id in sampled_batch and False: 
				path = f'save/examples/specificAccuracy{batch_id}.png'
				# want to generate plot of accuracy metrics vs time
				num_cols = 1
				num_rows = 1
				fig, axss = plt.subplots(
					num_rows, num_cols, figsize=(num_cols * 18, num_rows * 3), squeeze=False
				)
				axs = axss.flatten()

				key = (2, 5)
				inf = rawInf[key]
				timestepLabels = list(range(len(inf)))
		
				helpProbs = [x[0] for x in inf]
				hurtProbs = [x[1] for x in inf]

				label = str(key[0] * key[1]) + " particles"
				axs[0].plot(timestepLabels, helpProbs, label='Helping')
				axs[0].plot(timestepLabels, hurtProbs, label='Hindering')

				seek_conflict_label = "Helping"
				if seek_conflict:
					seek_conflict_label = "Hindering"

				axs[0].set_ylabel("Probability", fontsize=18)
				axs[0].set_xlabel("Time Step", fontsize=18)
				axs[0].legend(fontsize=15)

				utils.general.save_fig(fig, path, tight_layout_kwargs={"rect": [0, 0, 1, 1]})




				tmp_dir = utils.general.get_tmp_dir()
				img_paths  = []
				img_gui_paths = []
				save_dir = f"/om2/user/kunaljha/RecursiveReasoning{L1_model_dir}"
				#save_dir = local_L1_model_dir
				gif_path = f'save/examples/specificAccuracy{batch_id}.gif'

				L1_conflict_str = "Hindering" if bool(conflict_int_cpu) else "Helping"

				for timestep, state in enumerate(states_raw):
					img_path = f"{save_dir}{tmp_dir}/{timestep}.png"
					img_path_gui = f"{save_dir}{tmp_dir}/{timestep}_gui.png"

					plot_L2_snapshot(img_path, 
						img_path_gui,
						state, 
						None,
						L1_actions_raw[timestep],
						None,
						L2_actions_raw[timestep],
						{"other_agent_seek_conflict": rawInf["is"][timestep], "other_agent_current_belief": None, "prev_other_agent_seek_conflict": []},
						L1_conflict_str)
					img_paths.append(img_path)
					img_gui_paths.append(img_path_gui)
				utils.general.make_gif(img_paths, gif_path, 3)
				shutil.rmtree(save_dir + tmp_dir)


			num_predictions += len(states_raw)
			num_batches += 1
			if num_batches % 2 == 0:
				summarizeAccuracy()
			# if num_batches % 1 == 0:
			# 	summarizeAccuracy()

	summarizeAccuracy()


	randomParticles = {}
	randomPartStd = {}
	randomKLs = {}
	randomKLStd = {}
	particles = {}
	particlesStd = {}
	pKLs = {}
	pKLStd = {}
	for key, accuracies in action_accuracy.items():
		if key == 'is' or key == 'nn':
			continue
		if len(accuracies) == 0:
			continue
		acc = np.mean(accuracies)
		std = np.std(accuracies)

		klStd = np.std(action_kl[key])
		kl = np.mean(action_kl[key]) 

		L2_sample, L1_sample = key

		if L2_sample == 'random':
			if L1_sample in randomParticles:  # get average of common particles different layers
				randomParticles[L1_sample *  100 / 90] = (randomParticles[L1_sample *  100 / 90] + acc) / 2
				randomKLs[L1_sample * 100/ 90] = (randomKLs[L1_sample *  100 / 90] + kl) / 2
				randomKLStd[L1_sample *  100 / 90] = (randomKLStd[L1_sample *  100 / 90] + klStd) / 2
				randomPartStd[L1_sample *  100 / 90] = (randomPartStd[L1_sample *  100 / 90] + std) / 2
			else:
				randomParticles[L1_sample *  100 / 90] = acc
				randomKLs[L1_sample *  100 / 90] = kl
				randomPartStd[L1_sample *  100 / 90] = std 
				randomKLStd[L1_sample *  100 / 90] = klStd
		else:
			k = (L2_sample * L1_sample)* 100 / 90  # show percent of full IS
			if k in particles:
				particles[k] = (particles[k] + acc) / 2
				pKLs[k] = (pKLs[k] + kl) / 2
				particlesStd[k] = (particlesStd[k] + std) / 2
				pKLStd[k] = (pKLStd[k] + klStd) / 2
			else:
				particles[k] = acc
				pKLs[k] = kl
				particlesStd[k] = std 
				pKLStd[k] = klStd



	keys = []

	vals = []
	kls = []
	stds = []
	klstds = []

	rvals = []
	rkls = []
	rstds = []
	rklstds = []
	for k, v in particles.items():
		keys.append(k)
		vals.append(v)
		stds.append(particlesStd[k] / 10)
		kls.append(pKLs[k])
		klstds.append(pKLStd[k] / 10)


		# rvals.append(randomParticles[k])
		# rstds.append(randomPartStd[k] / 10)
		# rkls.append(randomKLs[k])
		# rklstds.append(randomKLStd[k] / 10)

	vals = np.array(vals)
	kls = np.array(kls)
	stds = np.array(stds)
	klstds = np.array(klstds)
	rvals = np.array(rvals)
	rkls = np.array(rkls)
	rklstds = np.array(rklstds)
	rstds = np.array(rstds)

	nnVals = np.array([np.mean(action_accuracy["nn"])] * len(keys))
	nnStds = np.array([np.std(action_accuracy["nn"]) / 10] * len(keys)) 
	isVals = np.array([np.mean(action_accuracy['is'])] * len(keys))
	isStds = np.array([np.std(action_accuracy['is']) / 10] * len(keys))
	nnKL = np.array([np.mean(action_kl['nn'])] * len(keys))
	nnKLStd = np.array([np.std(action_kl['nn']) / 10] * len(keys))

	uniformKL = np.array([np.mean(action_kl['uniform'])] * len(keys))
	uniformKLStd = np.array([np.std(action_kl['uniform']) / 10] * len(keys))


	labels = ["Ours", "Ours w/o NN", "ToMnet", "EI"]
	accList = [vals, rvals, nnVals, isVals]
	errList = [stds, rstds, nnStds, isStds]
	for i, acc in enumerate(accList):
		print(f"Accuracy {labels[i]}: {acc}; Errors: {errList[i]}")

	labels = ["Ours", "Ours w/o NN", "ToMnet", "Uniform"]
	klList = [kls, rkls, nnKL, uniformKL]
	errList = [klstds, rklstds, nnKLStd, uniformKLStd]
	for i, acc in enumerate(accList):
		print(f"KL {labels[i]}: {acc}; Errors: {errList[i]}")

	exit(0)


	# # accuracy and kl divergence plots
	# num_cols = 2
	# num_rows = 1
	# fig, axss = plt.subplots(
	# 	num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), squeeze=False
	# )
	# axs = axss.flatten()

	# ax = axs[0]
	# ax.plot(keys, isVals, linestyle='dashdot', color='gold', label='EI (72 particles)') 
	# ax.fill_between(keys, isVals-isStds, isVals+isStds, alpha=0.15, color='gold')

	# ax.plot(keys, nnVals, label='NN', color = 'red', linestyle='dashed') 
	# ax.fill_between(keys, nnVals-nnStds, nnVals+nnStds, alpha=0.15, color='red')

	# ax.scatter(keys, vals, color='blue')
	# ax.plot(keys, vals, label='Our Approach w/ NN', color='blue')
	# ax.fill_between(keys, vals-stds, vals+stds, alpha=0.15, color='blue')

	# ax.scatter(keys, rvals, color='green')
	# ax.plot(keys, rvals, label='Our Approach w/o NN', color='green')
	# ax.fill_between(keys, rvals-rstds, rvals+rstds, alpha=0.15, color='green')
	# ax.set_ylabel("Accuracy")
	# ax.set_xlabel("% of hypothesis space evaluated")
	# ax.legend()

	# ax = axs[1]
	# ax.plot(keys, nnKL, label='NN', color = 'red', linestyle='dashed') 
	# ax.fill_between(keys, nnKL-nnKLStd, nnVals+nnKLStd, alpha=0.15, color='red')

	# ax.scatter(keys, kls, color='blue')
	# ax.plot(keys, kls, label='Our Approach w/ NN', color='blue')
	# ax.fill_between(keys, kls-klstds, kls+klstds, alpha=0.15, color='blue')

	# ax.scatter(keys, rkls, color='green')
	# ax.plot(keys, rkls, label='Our Approach w/o NN', color='green')
	# ax.fill_between(keys, rkls-rklstds, rkls+rklstds, alpha=0.15, color='green')

	# ax.plot(keys, uniformKL, color='purple', label='Uniform', linestyle='dashed')
	# ax.fill_between(keys, uniformKL-uniformKLStd, uniformKL+uniformKLStd, alpha=0.15, color='purple')

	# ax.set_ylabel("KL Divergence")
	# ax.set_xlabel("% of hypothesis space evaluated")
	# ax.legend()

	# utils.general.save_fig(fig, f'save/examples/acc_kl_plots.pdf', tight_layout_kwargs={"rect": [0, 0, 1, 1]})




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
			axs[0].plot(xvals, yvals, label=label, color='red')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='red')
		elif key == 'is':
			label = 'EI'
			axs[0].plot(xvals, yvals, label=label, color='gold')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='gold')
		elif key == ("random", 10):
			label = f"Ours w/o NN"
			axs[0].plot(xvals, yvals, label=label, color='green')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='green')
		elif key != (2, 5):
			continue
		else:
			label = f"Ours"
			axs[0].plot(xvals, yvals, label=label, color='blue')
			axs[0].fill_between(xvals, yvals-yerrs, yvals+yerrs, alpha=0.15, color='blue')
		print(f"Key {key} Vals: {yvals}; Errs: {yerrs}")
	axs[0].set_ylabel("Accuracy", fontsize=18)
	axs[0].set_xlabel("Episode Progress (%)", fontsize=18)
	axs[0].tick_params(length=0)
	axs[0].tick_params(axis="x")
	axs[0].set_xticks(range(10))
	axs[0].set_xticklabels(x_ticks)
	axs[0].legend(fontsize=18)
	utils.general.save_fig(fig, f'save/examples/avgAccuracy.png', tight_layout_kwargs={"rect": [0, 0, 1, 1]})