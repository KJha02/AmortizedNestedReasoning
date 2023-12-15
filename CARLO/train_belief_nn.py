from car_utils.car_data import StateBeliefDataset, car_state_collate, joint_sa_tensor_to_state_action
from tqdm import tqdm
import torch
import argparse
import pdb
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from car_models.ToMnet_car import ToMnet_state_pred, ToMnet_exist_pred
import car_utils.network as network
import car_utils.general
import time
import random
import numpy as np
from test_scenario1 import percentCorrect

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def get_args_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=123, help="Random seed")
	parser.add_argument("--env-name", type=str, default="stateEstimation", help="Environment name")
	parser.add_argument("--save-dir", type=str, default="save", help="Save directory")
	parser.add_argument("--experiment-name", type=str, default="", help="Experiment name")
	parser.add_argument("--action-size", type=int, default=5, help="Action space size")
	parser.add_argument("--goal-size", type=int, default=3, help="Goal space size")
	parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
	parser.add_argument("--num-data-train", type=int, default=2, help="Training set size")
	parser.add_argument("--num-data-test", type=int, default=2, help="Testing set size")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
	parser.add_argument("--num-epochs", type=int, default=1000, help="Number of epochs")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--beta", type=float, default=0.01, help="How deterministic is the policy")
	parser.add_argument("--num-samples", type=int, default=3, help="How many particles are sampled in L0 IS_inference")
	parser.add_argument("--num-samples-L2", type=int, default=2, help="How many particles are sampled in L1 IS_inference")
	parser.add_argument("--sampled-actions", type=int, default=1, help="How actions to sample in policy")
	parser.add_argument("--lookAheadDepth", type=int, default=1, help="How far ahead to look when planning")

	parser.add_argument('--kunalDir', action='store_true')
	parser.set_defaults(kunalDir=False)
	
	return parser

def get_config_name(args):
	return (
		f"num_sampled_actions={args.sampled_actions},"
		f"lookAheadDepth={args.lookAheadDepth},"
		f"beta={args.beta}"
	)

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

def save_checkpoint(path, model, optimizer, stats, args=None):
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"stats": stats,
			"args": args,
		},
		path,
	)

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


def main(args):
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

	if args.kunalDir:
		BIG_STORAGE_DIR = "/scratch2/weka/tenenbaum/kunaljha/ReReason/RecursiveReasoning/"
	else:
		BIG_STORAGE_DIR = "../"

	state_model_save_dir = f"{BIG_STORAGE_DIR}CARLO/{args.save_dir}/{args.env_name}/{args.experiment_name}/{get_config_name(args)}"
	p = Path(state_model_save_dir)
	if not p.is_dir():
		p.mkdir(parents=True)
	dataset_dir = f"{BIG_STORAGE_DIR}data/{args.env_name}"

	train_dataset = StateBeliefDataset(
		beta=args.beta,
		num_data=args.num_data_train,
		seed=args.seed,
		dataset_dir=dataset_dir,
		train=True,
		device=device,
		num_inference_samples=args.num_samples,
		sampled_actions = args.sampled_actions,
		lookAheadDepth = args.lookAheadDepth,
		car1_exist_prior=0.65,
		car2_exist_prior=0.65
	)
	train_dataset.load()

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		collate_fn=car_state_collate,
		shuffle=True,
	)

	test_dataset = StateBeliefDataset(
		beta=args.beta,
		num_data=args.num_data_test,
		seed=args.seed,
		dataset_dir=dataset_dir,
		train=False,
		device=device,
		num_inference_samples=args.num_samples,
		sampled_actions = args.sampled_actions,
		lookAheadDepth = args.lookAheadDepth,
		car1_exist_prior=0.65,
		car2_exist_prior=0.65
	)
	test_dataset.load()

	test_dataloader = DataLoader(
		test_dataset,
		batch_size=args.batch_size,
		collate_fn=car_state_collate,
		shuffle=False,
	)


	stats = {"train_exist_losses": [], "test_exist_losses": [], "train_losses":[], 
	"train_telemetry_losses": [], "test_telemetry_losses": [], "test_losses":[],}


	state_model, state_optimizer = init_statePred(args, device)
	exist_model, exist_optimizer = init_existPred(args, device)

	kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
	mse_loss = torch.nn.MSELoss()

	scaler1 = torch.cuda.amp.GradScaler()
	scaler2 = torch.cuda.amp.GradScaler()

	min_test_loss = float("inf")
	min_test_kl_loss = float("inf")
	min_test_mse_loss = float("inf")
	for epoch_id in tqdm(range(args.num_epochs)):
		# training run
		temp_L = 0
		temp_kl = 0
		temp_mse = 0
		numBatches = 0
		state_model.train()
		exist_model.train()
		for batch_id, batch in enumerate(train_dataloader):
			state_actions, exist_tensors, telemetry_tensors, gt_cars = batch
			
			lens = torch.LongTensor([s.shape[0] for s in state_actions]).cpu()
			with torch.cuda.amp.autocast():
				log_prob_exist = exist_model(state_actions, lens)
				telemetry_pred = state_model(state_actions, lens)
				
				# handling whether car exists
				exist_target = torch.stack(exist_tensors, dim=0)  # fix this to be 0 for cars that don't exist
				exist_target = exist_target.view(exist_target.shape[0] * exist_target.shape[1], exist_target.shape[2])
				klDiv = kl_loss(log_prob_exist, exist_target)


				# handling telemetry of car
				mask = torch.stack(gt_cars, dim=0)  
				mask = mask.view(mask.shape[0] * mask.shape[1]) # (num timesteps x 16, 1)

				telemetry_target = torch.stack(telemetry_tensors, dim=0)
				telemetry_target = telemetry_target.view(telemetry_target.shape[0] * telemetry_target.shape[1], telemetry_target.shape[2])
				
				# apply mask
				telemetry_pred = telemetry_pred * mask.unsqueeze(1)
				telemetry_target = telemetry_target * mask.unsqueeze(1)

				mse = mse_loss(telemetry_pred, telemetry_target) 
			
			network.update_network(mse.double(), state_optimizer, scaler=scaler1, model=state_model)
			network.update_network(klDiv.double(), exist_optimizer, scaler=scaler2, model=exist_model)
			

			klDiv_val = klDiv.item()
			mse_val = mse.item()
			temp_L += klDiv_val + mse_val  # how is the joint loss declining
			temp_kl += klDiv_val
			temp_mse += mse_val
			numBatches += 1

		avgLoss = temp_L / numBatches
		avgKLLoss = temp_kl / numBatches
		avgMSELoss = temp_mse / numBatches
		stats["train_losses"].append(avgLoss)
		stats["train_exist_losses"].append(avgKLLoss)
		stats["train_telemetry_losses"].append(avgMSELoss)


		# testing run
		temp_L = 0
		numBatches = 0
		state_model.eval()
		exist_model.eval()
		avgAccuracy = 0.0
		for batch_id, batch in enumerate(test_dataloader):
			state_actions, exist_tensors, telemetry_tensors, gt_cars = batch
			lens = torch.LongTensor([s.shape[0] for s in state_actions]).cpu()
			with torch.cuda.amp.autocast():
				log_prob_exist = exist_model(state_actions, lens)
				telemetry_pred = state_model(state_actions, lens)
				
				# handling whether car exists
				exist_target = torch.stack(exist_tensors, dim=0)  # fix this to be 0 for cars that don't exist
				exist_target = exist_target.view(exist_target.shape[0] * exist_target.shape[1], exist_target.shape[2])
				klDiv = kl_loss(log_prob_exist, exist_target)


				# handling telemetry of car
				mask = torch.stack(gt_cars, dim=0)  
				mask = mask.view(mask.shape[0] * mask.shape[1]) # (num timesteps x 16, 1)

				telemetry_target = torch.stack(telemetry_tensors, dim=0)
				telemetry_target = telemetry_target.view(telemetry_target.shape[0] * telemetry_target.shape[1], telemetry_target.shape[2])
				
				# apply mask
				telemetry_pred = telemetry_pred * mask.unsqueeze(1)
				telemetry_target = telemetry_target * mask.unsqueeze(1)

				mse = mse_loss(telemetry_pred, telemetry_target) 

			klDiv_val = klDiv.item()
			mse_val = mse.item()
			temp_L += klDiv_val + mse_val  # how is the joint loss declining
			temp_kl += klDiv_val
			temp_mse += mse_val
			numBatches += 1

		avgLoss = temp_L / numBatches
		avgKLLoss = temp_kl / numBatches
		avgMSELoss = temp_mse / numBatches
		stats["test_losses"].append(avgLoss)
		stats["test_exist_losses"].append(avgKLLoss)
		stats["test_telemetry_losses"].append(avgMSELoss)

		if avgMSELoss < min_test_mse_loss:
			min_test_mse_loss = avgMSELoss
			save_checkpoint(f"{state_model_save_dir}/checkpoints/best_acc_state.pik", state_model, state_optimizer, stats, args)
		if avgKLLoss < min_test_kl_loss:
			min_test_kl_loss = avgKLLoss
			save_checkpoint(f"{state_model_save_dir}/checkpoints/best_acc_exist.pik", exist_model, exist_optimizer, stats, args)

		if epoch_id % 10 == 0:
			print("\n -----")
			print(f"Avg Train Loss Epoch {epoch_id}: {stats['train_losses'][-1]} ; Avg Test Loss Epoch {epoch_id}: {stats['test_losses'][-1]}")
			print(f"Avg Train KL Loss Epoch {epoch_id}: {stats['train_exist_losses'][-1]} ; Avg Test KL Loss Epoch {epoch_id}: {stats['test_exist_losses'][-1]}")
			print(f"Avg Train MSE Loss Epoch {epoch_id}: {stats['train_telemetry_losses'][-1]} ; Avg Test MSE Loss Epoch {epoch_id}: {stats['test_telemetry_losses'][-1]}")
			
			# Plot stats
			fig, axss = plt.subplots(
				1, 2, figsize=(2 * 8, 2 * 6), squeeze=False
			)
			axs = axss.flatten()

			ax = axs[0]
			ax.plot(stats["train_exist_losses"], label="Train")
			ax.plot(stats["test_exist_losses"], label="Test")
			ax.legend()
			ax.set_xlabel('Epoch')
			ax.set_ylabel('Loss')
			ax.set_title('KL Divergence Loss over Belief of Car Existence')

			ax = axs[1]
			ax.plot(stats["train_telemetry_losses"], label="Train")
			ax.plot(stats["test_telemetry_losses"], label="Test")
			ax.legend()
			ax.set_xlabel('Epoch')
			ax.set_ylabel('Loss')
			ax.set_title('MSE Loss over Car Telemetry Data')

			car_utils.general.save_fig(fig, f"{state_model_save_dir}/stats.png")
			



if __name__ == "__main__":
	parser = get_args_parser()
	args = parser.parse_args()
	main(args)

		
		