import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import tqdm
from pathlib import Path
import envs.construction_sample
from utils.construction_data import block_pair_utilities_to_desire_int
from utils.construction_data import get_rollout_gt_inference
import pdb
import models
import test_reasoning_about_construction_L0
from test_reasoning_about_construction_L0 import particle_inference, online_importance_sampling
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

from test_construction_agent_L0 import plot_L0_snapshot

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
cuda = torch.cuda.is_available()
if cuda:
	torch.cuda.manual_seed(123)
	device = "cuda"
else:
	device = "cpu"

num_samples = int(sys.argv[1])
num_possible_block_pairs = 45


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

def avgPredictionAccuracy(prediction, target):
	'''
	Returns average number of instances in which max probability proposal was true utility
	'''
	return np.sum(np.argmax(prediction, axis=-1) == target) / len(prediction)


# predicting every timestep
last = 0


BIG_STORAGE_DIR = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning"
local_save_dir = "/save/construction/30.0kDat_smallerModel_128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/"


def get_config_name():
	return (
		f"num_colored_block_locations=10,"
		f"num_possible_block_pairs=45,"
		f"beta=0.01"
	)

save_dir = BIG_STORAGE_DIR + local_save_dir
p = Path(save_dir)
if not p.is_dir():
	p.mkdir(parents=True)

num_train_data = 10000
dataset_dir = f"{BIG_STORAGE_DIR}/data/construction/"
print(dataset_dir)
dataset_train = construction_data.ReasoningAboutL0Dataset(
	num_colored_blocks=10,
	num_possible_block_pairs=45,
	num_rows=20,
	num_cols=20,
	beta=0.01,
	utility_mode="ranking",
	num_data=num_train_data,
	dataset_dir=dataset_dir,
	train=True,
	seed=123,
	device=device,
)
dataset_train.load()

train_dataloader = DataLoader(
	dataset_train,
	batch_size=128,
	collate_fn=construction_data.my_collate_last if last else construction_data.my_collate,
	shuffle=False,
)


# loading in model
model_dir = f"{save_dir}checkpoints/best_acc.pik"
model, optimizer, stats, args = train_construction_desire_pred.load_checkpoint(model_dir, device)
model.eval()

sampled_batch = random.sample(range(0, len(train_dataloader)), 5)


i = 0


avgKL_gt_is_nn = 0
avgKL_gt_nn = 0

avgAcc_is_nn = 0
avgAcc_is = 0
avgAcc_nn = 0

for batch_id, batch in enumerate(train_dataloader):
	# if i in sampled_batch:
	if True:
		# gif_path = f"{save_dir}sample_gifs/{i}_train.gif"
		print(f"Processing training batch {batch_id}")

		states, actions, desires, IS_inference = batch[0], batch[1], batch[2], batch[3]
		lens = torch.LongTensor([s.shape[0] for s in states]).to('cpu')
		actions_2d = [
			utils.network.expand_batch(a, (20, 20, 6))
			for a in actions
		]

		# with torch.cuda.amp.autocast():
		log_prob = model(states, actions_2d, lens, last=last)  # the final inference prediction for the model
		nn_inferences = torch.softmax(log_prob, 1).cpu().detach().numpy()
		nn_inferences /= nn_inferences.sum(axis=1, keepdims=True)


		state_tensor_rollouts = states[0]
		states_raw = [construction_data.state_tensor_to_state(s) for s in state_tensor_rollouts]
		action_tensor_rollouts = actions[0]
		actions_raw = [construction_data.action_tensor_to_action(a) for a in action_tensor_rollouts]
		rollout_desire_int = desires[0]
		initial_state = states_raw[0]
		colored_block_desire = construction_data.desire_int_to_utilities(int(rollout_desire_int), num_possible_block_pairs)
		rollout_env = envs.construction.ConstructionEnv(initial_state, colored_block_desire)

		nn_rollout_probs = nn_inferences[:len(states_raw)]

		# full_is_nn_inference = online_importance_sampling(rollout_env, 
		# 	states_raw, 
		# 	actions_raw,
		# 	colored_block_utilities_proposal_probss=nn_rollout_probs,
		# 	num_samples=45)

		is_nn_inferences = online_importance_sampling(rollout_env, 
			states_raw, 
			actions_raw,
			colored_block_utilities_proposal_probss=nn_rollout_probs,
			num_samples=num_samples)

		is_nn_probs = []
		for j in range(len(is_nn_inferences)):
			posterior_belief = test_reasoning_about_construction_L0.get_posterior(is_nn_inferences[j][0], is_nn_inferences[j][1], sort_posterior=False)
			posterior_distrib = [0.0] * num_possible_block_pairs
			for p in posterior_belief:  # doing this to preserve order in predictions
				inferred_util_idx = block_pair_utilities_to_desire_int(dict(p[0]), num_possible_block_pairs)  # going from utility to index for consistency
				inferred_util_pred = p[1]  # what is the actual probability assigned to this belief
				posterior_distrib[inferred_util_idx] = inferred_util_pred
			is_nn_probs.append(posterior_distrib)


		# full_is_nn_probs = []
		# for j in range(len(full_is_nn_inference)):
		# 	posterior_belief = test_reasoning_about_construction_L0.get_posterior(full_is_nn_inference[j][0], full_is_nn_inference[j][1], sort_posterior=False)
		# 	posterior_distrib = [0.0] * num_possible_block_pairs
		# 	for p in posterior_belief:  # doing this to preserve order in predictions
		# 		inferred_util_idx = block_pair_utilities_to_desire_int(dict(p[0]), num_possible_block_pairs)  # going from utility to index for consistency
		# 		inferred_util_pred = p[1]  # what is the actual probability assigned to this belief
		# 		posterior_distrib[inferred_util_idx] = inferred_util_pred
		# 	full_is_nn_probs.append(posterior_distrib)




		# divergence between IS and Online IS + NN
		avgKL_gt_is_nn += avg_KL(IS_inference[:len(is_nn_probs)], is_nn_probs)
		# divergence between IS and NN
		avgKL_gt_nn += avg_KL(IS_inference[:len(nn_inferences)], nn_inferences)

		# accuracy for IS
		desire_int_cpu = rollout_desire_int.item()
		
		# acc_full_is_nn = avgPredictionAccuracy(full_is_nn_probs, desire_int_cpu)
		# is_sampled = get_rollout_gt_inference(states_raw, actions_raw, desire_int_cpu, num_possible_block_pairs, num_samples=num_samples)

		acc_is = avgPredictionAccuracy(IS_inference[:len(nn_inferences)], desire_int_cpu)
		avgAcc_is += acc_is
		acc_nn = avgPredictionAccuracy(nn_rollout_probs, desire_int_cpu)
		avgAcc_nn += acc_nn

		acc_is_nn = avgPredictionAccuracy(is_nn_probs, desire_int_cpu)
		avgAcc_is_nn += acc_is_nn

		# if acc_full_is_nn < acc_is_nn:
		# 	full_is_nn_inference = online_importance_sampling(rollout_env, 
		# 		states_raw, 
		# 		actions_raw,
		# 		colored_block_utilities_proposal_probss=nn_rollout_probs,
		# 		num_samples=45)


		# tmp_dir = utils.general.get_tmp_dir()
		# img_paths  = []
		# img_gui_paths = []
		# for timestep, state in enumerate(states_raw):
		# 	img_path = f"{save_dir}{tmp_dir}/{timestep}.png"
		# 	img_path_gui = f"{save_dir}{tmp_dir}/{timestep}_gui.png"

		# 	plot_L0_snapshot(img_path, 
		# 		img_path_gui,
		# 		state, 
		# 		colored_block_desire, 
		# 		is_nn_probs[timestep], 
		# 		IS_inference[timestep],
		# 		nn_rollout_probs[timestep],
		# 		rollout_desire_int)
		# 	img_paths.append(img_path)
		# 	img_gui_paths.append(img_path_gui)
		# utils.general.make_gif(img_paths, gif_path, 3)
		# shutil.rmtree(save_dir + tmp_dir)

	i += 1

avgKL_gt_is_nn /= len(train_dataloader)
avgKL_gt_nn /= len(train_dataloader)
print(f"Using {num_samples} samples:")
print(f"\nApproximate KL divergence between IS inference and Online IS + NN in training is {avgKL_gt_is_nn}")
print(f"Approximate KL divergence between IS inference and NN in training is {avgKL_gt_nn}")


avgAcc_is /= len(train_dataloader)
avgAcc_nn /= len(train_dataloader)
avgAcc_is_nn /= len(train_dataloader)
print(f"\nApproximate avg accuracy for IS in training is {avgAcc_is}")
print(f"Approximate avg accuracy for NN in training is {avgAcc_nn}")
print(f"Approximate avg accuracy for Online IS + NN in training with {num_samples} samples is {avgAcc_is_nn}\n")

dataset_test = construction_data.ReasoningAboutL0Dataset(
	num_colored_blocks=10,
	num_possible_block_pairs=45,
	num_rows=20,
	num_cols=20,
	beta=0.01,
	utility_mode="ranking",
	num_data=int(num_train_data * 0.2),
	dataset_dir=dataset_dir,
	train=False,
	seed=123,
	device=device,
)
dataset_test.load()

test_dataloader = DataLoader(
	dataset_test,
	batch_size=128,
	collate_fn=construction_data.my_collate_last if last else construction_data.my_collate,
	shuffle=False,
)
print("Loaded test data")


sampled_batch = random.sample(range(0, len(test_dataloader)), 5)
i = 0
num_gifs = 5
avgKL_gt_is_nn = 0
avgKL_gt_nn = 0

avgAcc_is_nn = 0
avgAcc_is = 0
avgAcc_nn = 0
for batch_id, batch in enumerate(test_dataloader):
	# if i in sampled_batch:
	if True:
		# gif_path = f"{save_dir}sample_gifs/{i}_test.gif"
		print(f"Processing testing batch {batch_id}")

		states, actions, desires, IS_inference = batch[0], batch[1], batch[2], batch[3]
		lens = torch.LongTensor([s.shape[0] for s in states]).to('cpu')
		actions_2d = [
			utils.network.expand_batch(a, (20, 20, 6))
			for a in actions
		]

		# with torch.cuda.amp.autocast():
		log_prob = model(states, actions_2d, lens, last=last)  # the final inference prediction for the model
		nn_inferences = torch.softmax(log_prob, 1).cpu().detach().numpy()
		nn_inferences /= nn_inferences.sum(axis=1, keepdims=True)


		# IS_inference, sampled_rollout_idx, sampled_inference_idxes = get_gt_inference(states, actions, desires, last=last, return_sample_idx=True)  # the ground truth importance sampling final inference

		state_tensor_rollouts = states[0]  # arbitrarily pick the first rollout in the batch
		states_raw = [construction_data.state_tensor_to_state(s) for s in state_tensor_rollouts]
		action_tensor_rollouts = actions[0]
		actions_raw = [construction_data.action_tensor_to_action(a) for a in action_tensor_rollouts]
		rollout_desire_int = desires[0]
		initial_state = states_raw[0]
		colored_block_desire = construction_data.desire_int_to_utilities(int(rollout_desire_int), num_possible_block_pairs)
		rollout_env = envs.construction.ConstructionEnv(initial_state, colored_block_desire)

		nn_rollout_probs = nn_inferences[:len(states_raw)]
		is_nn_inferences = online_importance_sampling(rollout_env, 
			states_raw, 
			actions_raw,
			colored_block_utilities_proposal_probss=nn_rollout_probs,
			num_samples=num_samples)

		is_nn_probs = []
		for j in range(len(is_nn_inferences)):
			posterior_belief = test_reasoning_about_construction_L0.get_posterior(is_nn_inferences[j][0], is_nn_inferences[j][1], sort_posterior=False)
			posterior_distrib = [0.0] * num_possible_block_pairs
			for p in posterior_belief:  # doing this to preserve order in predictions
				inferred_util_idx = block_pair_utilities_to_desire_int(dict(p[0]), num_possible_block_pairs)  # going from utility to index for consistency
				inferred_util_pred = p[1]  # what is the actual probability assigned to this belief
				posterior_distrib[inferred_util_idx] = inferred_util_pred
			is_nn_probs.append(posterior_distrib)


		# divergence between IS and Online IS + NN
		avgKL_gt_is_nn += avg_KL(IS_inference[:len(is_nn_probs)], is_nn_probs)
		# divergence between IS and NN
		avgKL_gt_nn += avg_KL(IS_inference[:len(nn_inferences)], nn_inferences)

		# accuracy for IS
		desire_int_cpu = rollout_desire_int.item()

		is_sampled = get_rollout_gt_inference(states_raw, actions_raw, desire_int_cpu, num_possible_block_pairs, num_samples=num_samples)

		acc_is = avgPredictionAccuracy(is_sampled, desire_int_cpu)
		avgAcc_is += acc_is
		acc_nn = avgPredictionAccuracy(nn_rollout_probs, desire_int_cpu)
		avgAcc_nn += acc_nn
		acc_is_nn = avgPredictionAccuracy(is_nn_probs, desire_int_cpu)
		avgAcc_is_nn += acc_is_nn

		# if num_gifs > 0 and acc_nn > acc_is_nn:  # view speicifc examples where nn is better than is + nn
		# 	tmp_dir = utils.general.get_tmp_dir()
		# 	img_paths  = []
		# 	img_gui_paths = []
		# 	for timestep, state in enumerate(states_raw):
		# 		img_path = f"{save_dir}{tmp_dir}/{timestep}.png"
		# 		img_path_gui = f"{save_dir}{tmp_dir}/{timestep}_gui.png"

		# 		plot_L0_snapshot(img_path, 
		# 			img_path_gui,
		# 			state, 
		# 			colored_block_desire, 
		# 			is_nn_probs[timestep], 
		# 			IS_inference[timestep],
		# 			nn_rollout_probs[timestep],
		# 			rollout_desire_int)
		# 		img_paths.append(img_path)
		# 		img_gui_paths.append(img_path_gui)
		# 	utils.general.make_gif(img_paths, gif_path, 3)
		# 	shutil.rmtree(save_dir + tmp_dir)

		# 	num_gifs -= 1
	i += 1

avgKL_gt_is_nn /= len(test_dataloader)
avgKL_gt_nn /= len(test_dataloader)
print(f"\nApproximate KL divergence between IS inference and Online IS + NN in testing is {avgKL_gt_is_nn}")
print(f"Approximate KL divergence between IS inference and NN in testing is {avgKL_gt_nn}")


avgAcc_is /= len(test_dataloader)
avgAcc_nn /= len(test_dataloader)
avgAcc_is_nn /= len(test_dataloader)
print(f"\nApproximate avg accuracy for IS in testing is {avgAcc_is}")
print(f"Approximate avg accuracy for NN in testing is {avgAcc_nn}")
print(f"Approximate avg accuracy for Online IS + NN in testing with {num_samples} samples is {avgAcc_is_nn}\n")


