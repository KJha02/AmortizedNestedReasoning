import torch
import argparse
import pdb
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import tqdm
from pathlib import Path
import envs.construction_sample
import models
import test_reasoning_about_construction_L0
import utils
from utils import construction_data
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import time
import sys
from scipy.stats import entropy
import time
import torch.nn.functional as F
# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'




def get_args_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=123, help="Random seed")
	parser.add_argument("--env-name", type=str, default="construction", help="Environment name")
	parser.add_argument("--save-dir", type=str, default="save", help="Save directory")
	parser.add_argument("--experiment-name", type=str, default="", help="Experiment name")
	parser.add_argument(
		"--utility-mode",
		type=str,
		default="ranking",
		choices=["ranking", "top"],
		help="Representation of the agent's utility",
	)
	parser.add_argument(
		"--num-colored-block-locations", type=int, default=10, help="Number of colored_blocks on the map",
	)
	parser.add_argument(
		"--num-possible-block-pairs", type=int, default=45, help="Number of possible block pairs",
	)
	parser.add_argument("--num-rows", type=int, default=20, help="Height of the map")
	parser.add_argument("--num-cols", type=int, default=20, help="Width of the map")
	parser.add_argument("--action-size", type=int, default=6, help="Action space size")
	parser.add_argument("--num-channels", type=int, default=128, help="Number of channels in CNN")
	parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
	parser.add_argument("--num-data-train", type=int, default=10000, help="Training set size")
	parser.add_argument("--num-data-test", type=int, default=2000, help="Testing set size")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
	parser.add_argument("--num-epochs", type=int, default=300, help="Number of epochs")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--model-type", type=str, default="ToMnet_DesirePred", help="Model type")
	parser.add_argument("--last", type=int, default=1, help="Only output last prediction")
	parser.add_argument("--beta", type=float, default=0.01, help="How deterministic is the policy")
	parser.add_argument("--num-samples", type=int, default=5, help="How many particles are sampled in L0 IS_inference")
	parser.add_argument("--num-samples-L2", type=int, default=2, help="How many particles are sampled in L1 IS_inference")
	# parser.add_argument("--useBFS", type=bool, default=0, help="Use heuristic plan or BFS")
	parser.add_argument('--useBFS', action='store_true')
	parser.add_argument('--noBFS', dest='useBFS', action='store_false')
	parser.set_defaults(useBFS=False)
	parser.add_argument('--kunalDir', action='store_true')
	parser.add_argument('--no-kunal', dest='kunalDir', action='store_false')
	parser.set_defaults(kunalDir=False)

	parser.add_argument('--trainL1', action='store_true')
	parser.add_argument('--trainL0', dest='trainL1', action='store_false')
	parser.set_defaults(trainL1=False)

	parser.add_argument('--l0-model-dir', type=str, default='', help='Path to L0 inference model')
	return parser


def get_config_name(args):
	return (
		f"num_colored_block_locations={args.num_colored_block_locations},"
		f"num_possible_block_pairs={args.num_possible_block_pairs},"
		f"beta={args.beta}"
	)


def init(args, device):
	if args.utility_mode == "ranking":
		output_dim = envs.construction.get_num_rankings(args.num_colored_block_locations)
	elif args.utility_mode == "top":
		if args.num_possible_block_pairs != \
				envs.construction.get_num_rankings(args.num_colored_block_locations):
			raise ValueError(
				"Number of possible block pairs must be equal to number of colored block "
				"locations choose 2 in 'top' utility mode"
			)
		output_dim = args.num_possible_block_pairs
	# print("Output dimensions is:" + str(output_dim))
	if args.model_type == "ToMnet_DesirePred":
		model = models.ToMnet_DesirePred(
			state_dim=(2*args.num_colored_block_locations + 4,args.num_rows, args.num_cols, ),
			action_size=args.action_size,
			num_channels=args.num_channels,
			hidden_dim=args.hidden_dim,
			output_dim=output_dim,
			rank=0
		)
	else:
		raise ValueError("Invalid model type:", args.model_type)
	model.to(device)
	# pred_model = DDP(model, device_ids=[device], output_device=device)
	optimizer = optim.AdamW(model.parameters(), lr=args.lr)
	return model, optimizer


def init_actionPred(args, device):
	if args.utility_mode == "ranking":
		utility_dim= envs.construction.get_num_rankings(args.num_colored_block_locations)
	elif args.utility_mode == "top":
		if args.num_possible_block_pairs != \
				envs.construction.get_num_rankings(args.num_colored_block_locations):
			raise ValueError(
				"Number of possible block pairs must be equal to number of colored block "
				"locations choose 2 in 'top' utility mode"
			)
		utility_dim = args.num_possible_block_pairs
	action_model = models.ToMnet_DesirePred(
			state_dim=(3*args.num_colored_block_locations + 4,args.num_rows, args.num_cols, ),
			action_size=args.action_size,
			num_channels=args.num_channels,
			hidden_dim=args.hidden_dim,
			output_dim=2,
		)
	action_model.to(device)
	action_optimizer = optim.Adam(action_model.parameters(), lr=args.lr)
	return action_model, action_optimizer


def save_checkpoint(path, model, optimizer, stats, args=None):
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	if model.rank == 0:
		torch.save(
			{
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"stats": stats,
				"args": args,
			},
			path,
		)
		print(f"Saved checkpoint to {path}")

def load_checkpoint(path, device, num_tries=3, L1=False):
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
	if L1:
		model, optimizer = init_actionPred(args, device)
	else:
		model, optimizer = init(args, device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	stats = checkpoint["stats"]
	return model, optimizer, stats, args

def main(args, rank=0):
	if args.experiment_name == "":
		raise RuntimeError("Provide --experiment-name")
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	cuda = torch.cuda.is_available()
	if cuda:
		torch.cuda.manual_seed(args.seed)
		device = "cuda"
	else:
		device = "cpu"
	# device = rank

	save_dir = f"{args.save_dir}/{args.env_name}/{args.experiment_name}/{get_config_name(args)}"
	if args.kunalDir:
		BIG_STORAGE_DIR = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning"
		save_dir = f"{BIG_STORAGE_DIR}/{save_dir}"
	else:
		BIG_STORAGE_DIR = ""

	p = Path(save_dir)
	if not p.is_dir():
		p.mkdir(parents=True)

	# Init data
	# num_train_data = int(args.num_data_train * 0.8)
	num_train_data = args.num_data_train
	num_validation_data = args.num_data_train - num_train_data
	dataset_dir = f"{BIG_STORAGE_DIR}/data/{args.env_name}"

	if not args.trainL1:
		dataset_train = construction_data.ReasoningAboutL0Dataset(
			num_colored_blocks=args.num_colored_block_locations,
			num_possible_block_pairs=args.num_possible_block_pairs,
			num_rows=args.num_rows,
			num_cols=args.num_cols,
			beta=args.beta,
			utility_mode=args.utility_mode,
			num_data=num_train_data,
			dataset_dir=dataset_dir,
			train=True,
			seed=args.seed,
			device=device,
			last=args.last,
			useBFS=args.useBFS
		)
		dataset_train.load()
		print("Loaded train data")

		dataset_test = construction_data.ReasoningAboutL0Dataset(
			num_colored_blocks=10,
			num_possible_block_pairs=45,
			num_rows=args.num_rows,
			num_cols=args.num_cols,
			beta=args.beta,
			utility_mode=args.utility_mode,
			num_data=args.num_data_test,
			dataset_dir=dataset_dir,
			train=False,
			seed=args.seed,
			device=device,
			last=args.last,
			useBFS=bool(args.useBFS)
		)
		dataset_test.load()
		print("Loaded test data")

		train_dataloader = DataLoader(
			dataset_train,
			batch_size=args.batch_size,
			collate_fn=construction_data.my_collate_last if args.last else construction_data.my_collate,
			shuffle=False,
		)
		test_dataloader = DataLoader(
			dataset_test,
			batch_size=args.batch_size,
			collate_fn=construction_data.my_collate_last if args.last else construction_data.my_collate,
			shuffle=False,
		)

		model, optimizer = init(args, device)

	else:
		if not args.kunalDir and len(args.l0_model_dir) == 0:
			raise RuntimeError("Provide --l0-model-dir")
		elif args.kunalDir and len(args.l0_model_dir) == 0:
			L0_model_path = f"{BIG_STORAGE_DIR}/save/construction/30.0kDat_smallerModel_128dim_128chan_0.0001lr_128bSize/num_colored_block_locations=10,num_possible_block_pairs=45,beta=0.01/checkpoints/best_acc.pik"
		else:
			L0_model_path = f"{BIG_STORAGE_DIR}/{args.l0_model_dir}"

		L0_inference_model, optimizer, stats, L0_args = load_checkpoint(L0_model_path, device)
		L0_inference_model.eval()
		print("Loaded L0 model")

		L1_train_dataset = construction_data.ReasoningAboutL1Dataset(
			num_colored_blocks=args.num_colored_block_locations,
			num_possible_block_pairs=args.num_possible_block_pairs,
			num_rows=args.num_rows,
			num_cols=args.num_cols,
			beta=args.beta,
			utility_mode=args.utility_mode,
			num_data=num_train_data,
			dataset_dir=dataset_dir,
			train=True,
			seed=args.seed,
			device=device,
			last=args.last,
			saved_model_dir=L0_model_path,
			L0_inference_model=L0_inference_model,
			num_samples=args.num_samples,
			num_samples_L2=args.num_samples_L2,
			useBFS=args.useBFS
		)
		L1_train_dataset.load()
		print("Loaded L1 train data")

		L1_test_dataset = construction_data.ReasoningAboutL1Dataset(
			num_colored_blocks=args.num_colored_block_locations,
			num_possible_block_pairs=args.num_possible_block_pairs,
			num_rows=args.num_rows,
			num_cols=args.num_cols,
			beta=args.beta,
			utility_mode=args.utility_mode,
			num_data=args.num_data_test,
			dataset_dir=dataset_dir,
			train=False,
			seed=args.seed,
			device=device,
			last=args.last,
			saved_model_dir = L0_model_path,
			L0_inference_model=L0_inference_model,
			num_samples=args.num_samples,
			num_samples_L2=args.num_samples_L2,
			useBFS=args.useBFS
		)
		L1_test_dataset.load()
		print("Loaded L1 test data")


		train_dataloader = DataLoader(
			L1_train_dataset,
			batch_size=args.batch_size,
			collate_fn=construction_data.my_collate_last if args.last else construction_data.my_collate,
			shuffle=False,
			# num_workers=80,
			# sampler=DistributedSampler(dataset_train),
		)
		test_dataloader = DataLoader(
			L1_test_dataset,
			batch_size=args.batch_size,
			collate_fn=construction_data.my_collate_last if args.last else construction_data.my_collate,
			shuffle=False,
			# num_workers=80,
			# sampler=DistributedSampler(dataset_train),
		)

		# action_nll_loss = torch.nn.NLLLoss().to(device)
		# try:
		# 	model_path = f"{save_dir}/checkpoints/best_acc.pik"
		# 	model, optimizer, stats, args = load_checkpoint(model_path, device)
		# 	print("Loaded model")
		# except:
		# 	print("Could not load model, generating new one")
		# 	model, optimizer = init(args, device)
		model, optimizer = init_actionPred(args, device)

	# Init model

	# action_model, action_optimizer = init_actionPred(args, device)

	# Train
	NLL_loss = torch.nn.NLLLoss().to(device)
	kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
	scaler = torch.cuda.amp.GradScaler()


	


	stats = {"train_losses": [], "train_divergence":[], "train_threshold":[], "validation_losses": [], "validation_threshold":[], 
		"validation_divergence":[], "validation_accuracies":[], "action_train_losses": [], "action_test_losses": [], 
		"test_divergence":[], "test_threshold":[], "test_losses":[], "train_accuracies": [], "test_accuracies": [],
		"uniform_divergence_train": [], "uniform_divergence_test": [], "exact_accuracy_train": [], "exact_accuracy_test": [],
		"exact_train_threshold": [], "exact_test_threshold": [], "exact_final_train_accuracy": [], "exact_final_test_accuracy": []}


	# optimizer = optim.Adam(model.parameters(), lr=args.lr) # rewind learning rate

	# learning_schedule = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.01, gamma=0.95, cycle_momentum=False)

	best_acc = 0
	min_loss = float("inf")
	min_action_loss = float("inf")
	for epoch_id in tqdm.tqdm(range(args.num_epochs)):
		# train_dataloader.sampler.set_epoch(epoch_id)  # done to have shuffling in multiprocessing

		# if epoch_id < 30:
		# 	continue

		accuracy = 0
		exact_accuracy = 0
		exact_final_accuracy = 0
		exact_over_threshold = 0
		cnt = 0
		avgKLDivergence = 0
		avgUniformKLDivergence = 0
		totKLCnt = 0
		overThreshold = 0
		temp_L = 0
		temp_action_L = 0
		numBatches = 0
		model.train()
		# action_model.train()
		trainBatchLen = len(train_dataloader)
		data_considered = 0
		for batch_id, batch in enumerate(train_dataloader):
			# if data_considered >= (2 * len(L1_train_dataset) // 100):  # training on less data
			# if data_considered >= (10 * len(dataset_train) // 100):
			# 	data_considered = 0
			# 	break
			# else:
			# 	data_considered += len(batch[0])
			# t1 = time.time()
			print(f"Doing batch {batch_id+1} / {trainBatchLen}")
			states, actions, desires, IS_inference, final_avg_accuracy = batch[0], batch[1], batch[2], batch[3], batch[4] # extract precomputed GT IS inference
			exact_final_accuracy += final_avg_accuracy

			lens = torch.LongTensor([s.shape[0] for s in states]).cpu()
			actions_2d = [
				utils.network.expand_batch(a, (args.num_rows, args.num_cols, args.action_size))
				for a in actions
			]
			# t2 = time.time()
			# print(f"Time to load and prep inputs = {t2 - t1} seconds")

			# t1 = time.time()
			# pdb.set_trace()
			with torch.cuda.amp.autocast():
				log_prob = model(states, actions_2d, lens, last=args.last)  # the final inference prediction for the model
				# loss = NLL_loss(log_prob, desires)
				loss = kl_loss(log_prob, torch.tensor(IS_inference).to(log_prob.dtype).to(rank))
			# t2 = time.time()
			# print(f"Time to predict = {t2 - t1} seconds")
			# action_model_desires = []
			# currStart = 0
			# for s in states:  # creating a one hot encoding of desires
			# 	temp_desires = torch.zeros(len(s), args.num_possible_block_pairs)
			# 	des_int = desires[currStart].item()
			# 	for t in range(len(temp_desires)):
			# 		temp_desires[t][des_int] = 1
			# 	action_model_desires.append(temp_desires.float().to(device))
			# 	currStart = len(s)

			# desire_pred_2d = [
			# 	utils.network.expand_batch(d, (args.num_rows, args.num_cols, args.num_possible_block_pairs))
			# 	for d in action_model_desires
			# ]

			# action_prediction_log_prob = action_model(states, desire_pred_2d, lens, last=args.last)
			# action_actual = [torch.argmax(a, dim=-1) for a in actions]
			# action_actual = torch.cat(action_actual)
			# log_prob = action_model(states, actions_2d, lens, last=args.last)
			# action_loss = action_nll_loss(log_prob, action_actual)
			temp_action_L += 0
			# utils.network.update_network(action_loss.double(), action_optimizer)

			# IS_inference = get_gt_inference(states, actions, desires, last=args.last)  # the ground truth importance sampling final inference
			# t1 = time.time()
			# for i in range(len(log_prob)):
			# 	nn_pred_distrib = np.exp(log_prob[i].cpu().detach().numpy())
			# 	is_pred_distrib = IS_inference[i]
				
			# 	if nn_pred_distrib[desires[i]] >= 1/args.num_possible_block_pairs:
			# 		overThreshold += 1
			# 	if is_pred_distrib[desires[i]] >= 1/args.num_possible_block_pairs:
			# 		exact_over_threshold += 1


			# 	# num_zeros = is_pred_distrib.count(0)  # find number of zeros
			# 	# num_nonzeros = len(is_pred_distrib) - num_zeros
			# 	# if num_zeros > 0:
			# 	# 	noise_subtraction = (1e-6 * num_zeros) / num_nonzeros  # take away uniformly from non-zeros so we sum to 1
			# 	# 	for j, prob in enumerate(is_pred_distrib):
			# 	# 		if prob == 0:  # add some noise
			# 	# 			is_pred_distrib[j] = 1e-6
			# 	# 		else:
			# 	# 			is_pred_distrib[j] -= noise_subtraction
			# 	IS_inference[i] = is_pred_distrib  # modify original array for loss in future
			# 	# is_pred_distrib is an array with dimensions (num possible block pairs,)
			# 	# KL_diverge = sum(rel_entr(is_pred_distrib, nn_pred_distrib))  # calculate KL divergence of NN from IS distribution
			# 	KL_diverge = 0
			# 	avgKLDivergence += KL_diverge


			# 	# uniform_distrib = [1/len(nn_pred_distrib)] * len(nn_pred_distrib)
			# 	# uniform_KL_diverge = sum(rel_entr(is_pred_distrib, uniform_distrib))
			# 	uniform_KL_diverge = 0
			# 	avgUniformKLDivergence += uniform_KL_diverge
			# t2 = time.time()
			# print(f"Time to find predictions over uniform probability = {t2 - t1} seconds")


			totKLCnt += len(log_prob)

			# loss = NLL_loss(log_prob, desires)
			# t1 = time.time()
			
			# t2 = time.time()
			# print(f"Time to calculate loss = {t2 - t1} seconds")

			# if loss.item() < min_loss:
			#     min_loss = loss.item()
			#     save_checkpoint(f"{save_dir}/checkpoints/best_acc.pik", model, optimizer, stats, args)
			#     if accuracy <= 0.6:
			#         save_checkpoint(f"{save_dir}/checkpoints/sixty.pik", model, optimizer, stats, args)

			# t1 = time.time()
			pred = log_prob.argmax(-1)
			# accuracy += (pred == desires).float().sum()
			accuracy += 0
			cnt += len(desires)

			if epoch_id == 0:
				exact_pred = np.argmax(IS_inference, axis=-1)
				exact_accuracy += np.sum(exact_pred == desires.cpu().detach().numpy())
			# t2 = time.time()
			# print(f"Time to compute accuracy = {t2 - t1} seconds")

			# t1 = time.time()
			utils.network.update_network(loss.double(), optimizer, scaler=scaler, model=model)
			# t2 = time.time()
			# print(f"Time to do backprop= {t2 - t1} seconds")
			temp_L += loss.item()
			numBatches += 1

		# learning_schedule.step()  # cyclical step of learning rate

		stats["train_losses"].append(temp_L / numBatches)
		stats["action_train_losses"].append(temp_action_L / numBatches)
		# print("epoch {} training loss {}".format(epoch_id, temp_L/numBatches))
		accuracy /= cnt
		exact_accuracy /= cnt
		exact_final_accuracy /= numBatches
		avgKLDivergence /= totKLCnt
		avgUniformKLDivergence /= totKLCnt
		overThreshold /= cnt
		exact_over_threshold /= cnt

		
		assert accuracy <= 1.0, "Training accuracy is greater than 1"
		# if overThreshold > 0.8:
		#     sys.exit("Model suspiciously predicts correct ground truth utility at too high of a percentage")
		
		stats["exact_final_train_accuracy"].append(exact_final_accuracy)
		try:
			stats["train_accuracies"].append(accuracy.item())
		except:
			stats["train_accuracies"].append(accuracy)
		if epoch_id == 0:
			stats["exact_accuracy_train"].append(exact_accuracy)
		else:
			stats["exact_accuracy_train"].append(stats["exact_accuracy_train"][-1])
		stats["train_divergence"].append(avgKLDivergence)
		stats["uniform_divergence_train"].append(avgUniformKLDivergence)
		stats["train_threshold"].append(overThreshold)
		stats["exact_train_threshold"].append(exact_over_threshold)

		# print(f"Exact over threshold {exact_over_threshold}")

		# Compute test accuracy
		model.eval()
		# action_model.eval()
		# torch.no_grad()
		accuracy = 0
		exact_accuracy = 0
		exact_final_accuracy = 0
		overThreshold = 0
		exact_test_over_threshold = 0
		cnt = 0
		avgTestKLDivergence = 0
		avgUniformTestKLDivergence = 0
		totKLCnt = 0
		numBatches = 0
		temp_L = 0
		temp_action_L = 0
		min_loss = float("inf")
		testBatchLen = len(test_dataloader)
		for batch_id, batch in enumerate(test_dataloader):
			print(f"Doing batch {batch_id} / {testBatchLen}")
			states, actions, desires, IS_inference, final_avg_accuracy = batch[0], batch[1], batch[2], batch[3], batch[4]
			exact_final_accuracy += final_avg_accuracy
			lens = torch.LongTensor([s.shape[0] for s in states]).to('cpu')
			actions_2d = [
				utils.network.expand_batch(a, (args.num_rows, args.num_cols, args.action_size))
				for a in actions
			]

			# action_model_desires = []
			# currStart = 0
			# for s in states:  # creating a one hot encoding of desires
			# 	temp_desires = torch.zeros(len(s), args.num_possible_block_pairs)
			# 	des_int = desires[currStart].item()
			# 	for t in range(len(temp_desires)):
			# 		temp_desires[t][des_int] = 1
			# 	action_model_desires.append(temp_desires.float().to(device))
			# 	currStart = len(s)

			# desire_pred_2d = [
			# 	utils.network.expand_batch(d, (args.num_rows, args.num_cols, args.num_possible_block_pairs))
			# 	for d in action_model_desires
			# ]

			# log_prob = action_model(states, actions_2d, lens, last=args.last)
			# action_actual = [torch.argmax(a, dim=-1) for a in actions]
			# action_actual = torch.cat(action_actual)
			# action_loss = action_nll_loss(action_prediction_log_prob, action_actual)
			temp_action_L += 0


			with torch.cuda.amp.autocast():
				log_prob = model(states, actions_2d, lens, last=args.last)  # the final inference prediction for the model
				# loss = NLL_loss(log_prob, desires)
				loss = kl_loss(log_prob, torch.tensor(IS_inference).to(log_prob.dtype).to(rank))

			# for i in range(len(log_prob)):
			# 	nn_pred_distrib = np.exp(log_prob[i].cpu().detach().numpy())
			# 	is_pred_distrib = IS_inference[i]
				
			# 	if nn_pred_distrib[desires[i]] >= 1/args.num_possible_block_pairs:
			# 		overThreshold += 1
			# 	if is_pred_distrib[desires[i]] >= 1/args.num_possible_block_pairs:
			# 		exact_test_over_threshold += 1


			# 	# num_zeros = is_pred_distrib.count(0)  # find number of zeros
			# 	# num_nonzeros = len(is_pred_distrib) - num_zeros
			# 	# if num_zeros > 0:
			# 	# 	noise_subtraction = (1e-6 * num_zeros) / num_nonzeros  # take away uniformly from non-zeros so we sum to 1
			# 	# 	for j, prob in enumerate(is_pred_distrib):
			# 	# 		if prob == 0:  # add some noise
			# 	# 			is_pred_distrib[j] = 1e-6
			# 	# 		else:
			# 	# 			is_pred_distrib[j] -= noise_subtraction
			# 	IS_inference[i] = is_pred_distrib  # modify original array for loss in future
			# 	# is_pred_distrib is an array with dimensions (num possible block pairs,)
			# 	# KL_diverge = sum(rel_entr(is_pred_distrib, nn_pred_distrib))  # calculate KL divergence of NN from IS distribution
			# 	KL_diverge = 0
			# 	avgTestKLDivergence += KL_diverge


			# 	# uniform_distrib = [1/len(nn_pred_distrib)] * len(nn_pred_distrib)
			# 	# uniform_KL_diverge = sum(rel_entr(is_pred_distrib, uniform_distrib))
			# 	uniform_KL_diverge = 0
			# 	avgUniformTestKLDivergence += uniform_KL_diverge

			totKLCnt += len(log_prob)

			# loss = NLL_loss(log_prob, desires)
			
			
			pred = log_prob.argmax(-1)
			accuracy += 0
			temp_L += loss.item()
			numBatches += 1
			# accuracy += (pred == desires).float().sum()

			if epoch_id == 0:
				exact_pred = np.argmax(IS_inference, axis=-1)
				exact_accuracy += np.sum(exact_pred == desires.cpu().detach().numpy())

			# if loss.item() < min_loss and epoch_id % 100 == 0:
			# 	min_loss = loss.item()
			# 	save_checkpoint(f"{save_dir}/checkpoints/best_acc.pik", model, optimizer, stats, args)
			
			cnt += len(desires)


		if numBatches == 0:
			numBatches = 1
		if totKLCnt == 0:
			totKLCnt = 1
		if cnt == 0:
			cnt = 1

		stats["test_losses"].append(temp_L / numBatches)
		stats["action_test_losses"].append(temp_action_L / numBatches)
		# print("epoch {} test loss {}".format(epoch_id, temp_L / numBatches))
		accuracy /= cnt
		exact_accuracy /= cnt
		exact_final_accuracy /= numBatches
		avgTestKLDivergence /= totKLCnt
		avgUniformTestKLDivergence /= totKLCnt
		overThreshold /= cnt
		exact_test_over_threshold /= cnt
		try:
			assert accuracy <= 1.0
		except:
			print("Test accuracy is greater than 1")
			exit(2)

		stats["exact_final_test_accuracy"].append(exact_final_accuracy)
		try:
			stats["test_accuracies"].append(accuracy.item())
		except:
			stats["test_accuracies"].append(accuracy)
		stats["test_divergence"].append(avgTestKLDivergence)
		stats["uniform_divergence_test"].append(avgUniformTestKLDivergence)
		stats["test_threshold"].append(overThreshold)
		if epoch_id == 0:
			stats["exact_accuracy_test"].append(exact_accuracy)
		else:
			stats["exact_accuracy_test"].append(stats["exact_accuracy_test"][-1])
		stats["exact_test_threshold"].append(exact_test_over_threshold)

		# Save checkpoint with the best test accuracy
		if stats['test_accuracies'][-1] == max(stats['test_accuracies']):
			save_checkpoint(f"{save_dir}/checkpoints/best_acc.pik", model, optimizer, stats, args)

		if (epoch_id + 1) % 50 == 0:
			save_checkpoint(f"{save_dir}/checkpoints/epoch_{epoch_id + 1}.pik", model, optimizer, stats, args)

		# if (action_loss.data < min_action_loss) or (epoch_id % 50 == 0):
		# 	min_action_loss = action_loss.data
		# 	save_checkpoint(f"{save_dir}/checkpoints/best_action_acc.pik", action_model, action_optimizer, stats, args)
		print(
			f"epoch {epoch_id} train_loss {stats['train_losses'][-1]} "
			f"test_loss {stats['test_losses'][-1]}"
		)

		if epoch_id % 10 == 0:

			# Plot stats
			fig, axss = plt.subplots(
				2, 2, figsize=(2 * 6, 2 * 8), squeeze=False
			)
			axs = axss.flatten()
			# assert len(stats["train_accuracies"]) == epoch_id + 1
			ax = axs[0]
			ax.plot(stats["train_losses"], label="Train")
			ax.plot(stats["test_losses"], label="Test")
			ax.plot(stats["action_train_losses"], label="A Pred NLL Train")
			ax.plot(stats["action_test_losses"], label="A Pred NLL Test")
			# ax.plot(stats["validation_losses"], label="Validation")
			ax.set_ylabel("Loss")
			ax.set_xlabel("Epoch")
			ax.legend()
			
			ax2 = axs[1]
			ax2.plot(stats["train_accuracies"], label="Train")
			ax2.plot(stats["exact_accuracy_train"], label="Exact Train")
			ax2.plot(stats["test_accuracies"], label="Test")
			ax2.plot(stats["exact_accuracy_test"], label="Exact Test")
			ax2.plot(stats["exact_final_train_accuracy"], label="Exact Train Final")
			ax2.plot(stats["exact_final_test_accuracy"], label="Exact Test Final")
			# ax2.plot(stats["validation_accuracies"], label="Validation")
			ax2.set_ylabel("Avg Accuracy (arg max)")
			ax2.set_xlabel("Epoch")
			ax2.legend()

			ax3 = axs[2]
			ax3.plot(stats["train_divergence"], label="Train")
			ax3.plot(stats["uniform_divergence_train"], label="Uniform Train")
			ax3.plot(stats["test_divergence"], label="Test")
			ax3.plot(stats["uniform_divergence_test"], label="Uniform Test")
			# ax3.plot(stats["validation_divergence"], label="Validation")
			ax3.set_ylabel("Avg KL divergence")
			ax3.set_xlabel("Epoch")
			ax3.legend()

			ax4 = axs[3]
			ax4.plot(stats["train_threshold"], label="Train")
			ax4.plot(stats["exact_train_threshold"], label="Exact Train")
			ax4.plot(stats["test_threshold"], label="Test")
			ax4.plot(stats["exact_test_threshold"], label="Exact Test")
			# ax4.plot(stats["validation_threshold"], label="Validation")
			ax4.set_ylabel(f"Avg GT Predictions over 1/2 probability")
			ax4.set_xlabel("Epoch")
			ax4.legend()

			utils.general.save_fig(fig, f"{save_dir}/stats.png")


def get_gt_inference(states, actions, desires, last=True, return_entropy=False, return_sample_idx=False):
	'''

	Args:
		states: array of batch size n with each element being a state tensor
		actions: array of batch size n with each element being an action tensor
		desires: array of batch size n with each element being a desire tensor

	Returns: n-length array of probability distributions, each of length m, where
		n is the batch size
		m is the length of num possible block pairs

	'''
	# the following section determines the ground truth inferences based on importance sampling
	# only used for evaluation

	entropy_per_time = {}
	for i in range(20):
		entropy_per_time[i] = []
	avg_entropy_overall = 0

	num_steps = 0

	avg_entropy_per_ep = {}
	entrop_ranges = np.linspace(0.1, 2.0, 15)
	for e in entrop_ranges:
		avg_entropy_per_ep[e] = 0


	# used for visualizing specific snapshot
	sampled_rollout_idx = random.randint(0, len(states)-1)
	sample_inference_idxes = []

	IS_inferences = []
	for i, state_tensor_rollouts in enumerate(states):
		states_raw = [construction_data.state_tensor_to_state(s) for s in state_tensor_rollouts]
		action_tensor_rollouts = actions[i]
		actions_raw = [construction_data.action_tensor_to_action(a) for a in action_tensor_rollouts]
		rollout_desire_int = desires[i]
		initial_state = states_raw[0]
		colored_block_desire = construction_data.desire_int_to_utilities(int(rollout_desire_int), 3)
		rollout_env = envs.construction.ConstructionEnv(initial_state, colored_block_desire)

		# n-states length array containing [inference info, inference distribution] pairs
		all_inferences = test_reasoning_about_construction_L0.online_importance_sampling(rollout_env, states_raw, actions_raw)

		if last:
			# final inference
			final_posterior_belief = test_reasoning_about_construction_L0.get_posterior(all_inferences[-1][0], all_inferences[-1][1], sort_posterior=False)
			final_posterior_distrib = [i[1] for i in final_posterior_belief]  # isolate probabilities
			IS_inferences.append(final_posterior_distrib)
		else:
			# all inferences
			entropy_ep = 0
			for j in range(len(all_inferences)):
				posterior_belief = test_reasoning_about_construction_L0.get_posterior(all_inferences[j][0], all_inferences[j][1], sort_posterior=False)
				posterior_distrib = [k[1] for k in posterior_belief]
				IS_inferences.append(posterior_distrib)

				if return_sample_idx:
					if sampled_rollout_idx == i:
						sample_inference_idxes.append(num_steps)
					num_steps += 1

				if return_entropy:
					curr_ent = entropy(posterior_distrib, base=2)
					entropy_per_time[j].append(curr_ent)
					avg_entropy_overall += curr_ent
					entropy_ep += curr_ent
					num_steps += 1

			entropy_ep /= len(all_inferences)

			if return_entropy:
				i = 0
				while i < len(entrop_ranges):
					if entropy_ep > entrop_ranges[i]:
						i += 1
					else:
						break
				if i == len(entrop_ranges):
					i -=1
				avg_entropy_per_ep[entrop_ranges[i]] += 1

	if return_entropy:
		avg_entropy_overall /= num_steps

		avg_entropy_per_time = {}
		for step, vals in entropy_per_time.items():
			if len(vals) > 0:
				avg_entropy_per_time[step] = sum(vals) / len(vals)
			else:
				avg_entropy_per_time[step] = 0

		entrop_freqs = [avg_entropy_per_ep[e] for e in entrop_ranges]

	if return_entropy:
		return IS_inferences, avg_entropy_per_time, avg_entropy_overall, (entrop_ranges, entrop_freqs)
	if return_sample_idx:
		return IS_inferences, sampled_rollout_idx, sample_inference_idxes
	else:
		return IS_inferences


# def ddp_setup(rank: int, world_size: int):
# 	"""
# 	Args:
# 	   rank: Unique identifier of each process
# 	  world_size: Total number of processes
# 	"""
# 	os.environ["MASTER_ADDR"] = "localhost"
# 	os.environ["MASTER_PORT"] = "12355"
# 	init_process_group(backend="nccl", rank=rank, world_size=world_size)


# def ddp_main(rank, world_size, args):
# 	ddp_setup(rank, world_size)
# 	main(args, rank)
# 	destroy_process_group()


if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn', force=True)
	# torch.multiprocessing.set_sharing_strategy('file_system')
	torch.backends.cudnn.benchmark = True
	# world_size = torch.cuda.device_count()
	parser = get_args_parser()
	args = parser.parse_args()
	main(args)
	# mp.spawn(ddp_main, args=(world_size, args), nprocs=world_size)
