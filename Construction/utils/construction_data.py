import copy
import itertools
import pdb
import os
import torch
import torch.utils.data
import random
import pickle
import numpy as np
from pathlib import Path
import envs.construction_sample
import test_sample_construction_L0_rollout
from L1_rollout import default_L1_rollout
import test_reasoning_about_construction_L0
import envs.construction
import torch
import matplotlib.pyplot as plt
import utils.general
import shutil
from envs.construction import block2color
from torch.multiprocessing import set_start_method, Pool, Process
# from pathos.multiprocessing import ProcessingPool as Pool

def genInitialEnv(dataInfo):
	saved_model_dir, device, L0_inference_model, num_samples, num_samples_L2 = dataInfo
	new_rolls, env, desiress = default_L1_rollout(saved_model_dir=saved_model_dir, device=device, inference_model=L0_inference_model, num_samples=num_samples, num_samples_L2=num_samples_L2)
	# for i, r in enumerate(new_rolls):
	# 	rollout, done = r
	# 	try:
	# 		x = len(done)
	# 		pdb.set_trace()
	# 	except:
	# 		pass
	# 	rollouts.append((rollout, done))
	# 	desire_ints.append(desiress[i])
	return new_rolls, env, desiress


def genAdditionalEnv(inputArgs):
	env, dataInfo = inputArgs
	saved_model_dir, device, L0_inference_model, num_samples, num_samples_L2 = dataInfo

	colored_block_utilities_0 = envs.construction_sample.sample_block_pair_utilities(
	45, return_prob=False
	)
	colored_block_utilities_1 = envs.construction_sample.sample_block_pair_utilities(
	45, return_prob=False
	)
	# colored_block_utilities_1 = colored_block_utilities_0
	colored_block_utilities = {
		0: colored_block_utilities_0,
		1: colored_block_utilities_1,
	}
	env.colored_block_utilities = colored_block_utilities
	new_rolls, env, desiress = default_L1_rollout(env=environment, saved_model_dir=self.saved_model_dir, device=self.device, inference_model=self.L0_inference_model, num_samples=self.num_samples, num_samples_L2=self.num_samples_L2)
	# for i, r in enumerate(new_rolls):
	# 	rollout, done = r 
	# 	rollouts.append((rollout, done))
	# 	desire_ints.append(desiress[i])
	return new_rolls, env, desiress

class ReasoningAboutL1Dataset(torch.utils.data.Dataset):
	def __init__(
		self,
		utility_mode='ranking',
		beta=0.1,
		num_data=1000,
		seed=1,
		dataset_dir=None,
		saved_model_dir=None,
		L0_inference_model=None,
		train=True,
		device="cpu",
		num_rows=20,
		num_cols=20,
		num_colored_blocks=len(envs.construction.ALL_COLORED_BLOCKS),
		num_possible_block_pairs=len(envs.construction.ALL_BLOCK_PAIRS),
		num_samples=5,
		num_samples_L2=5,
		last=1,
		human=False,
		synthetic=False,
		useBFS=False
	):
		self.num_colored_blocks = num_colored_blocks
		self.synthetic = synthetic
		self.num_possible_block_pairs = num_possible_block_pairs
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.beta = beta
		self.utility_mode = utility_mode
		self.saved_model_dir = saved_model_dir
		self.seed = seed
		self.num_data = num_data
		self.dataset_dir = dataset_dir
		self.train = train
		self.device = device
		self.last = last == 1
		self.num_samples=num_samples
		self.num_samples_L2 = num_samples_L2
		self.human = human
		self.data = {}
		self.L0_inference_model = L0_inference_model
		self.useBFS = useBFS
		if self.L0_inference_model is not None:
			self.L0_inference_model.share_memory()

		if self.train:
			seed_offset = 0
		else:
			seed_offset = 1
		# random.seed(seed + seed_offset)
		# np.random.seed(seed + seed_offset)
		if dataset_dir:
			train_test_str = "train" if train else "test"
			self.dataset_path = (
				f"{dataset_dir}/L1_{num_colored_blocks}_{num_possible_block_pairs}_"
				f"{num_rows}_{num_cols}_{num_data}_{beta}_0_{train_test_str}_{utility_mode}.pik"
			)
			if synthetic:
				self.dataset_folder = dataset_dir  # data/easySynthetic/
			elif self.human:  # human dataset
				self.dataset_folder = f"{dataset_dir}/humanExp/"
			else:
				self.dataset_folder = (
					f"{dataset_dir}/L2_{num_samples_L2}Samples_L1_{num_samples}Samples_{num_colored_blocks}_{num_possible_block_pairs}_"
					f"{num_rows}_{num_cols}_{num_data}_{beta}_{train_test_str}_{utility_mode}/"
				)
			print(self.dataset_folder)

			if num_data == 10000 and train and False:
				self.dataset_folders = []
				for i in range(1000, 3000, 100):
					temp_data = i
					train_test_str = "train" if train else "test"
					temp_path =  (
						f"{dataset_dir}/L2_{num_samples_L2}Samples_L1_{num_samples}Samples_{num_colored_blocks}_{num_possible_block_pairs}_"
						f"{num_rows}_{num_cols}_{temp_data}_{beta}_{train_test_str}_{utility_mode}/"
					)
					self.dataset_folders.append(temp_path)
				# self.dataset_folders.append(self.dataset_folder)
			elif num_data == 2000 and not train and False:
				self.dataset_folders = []
				for i in range(1000, 3000, 100):
					temp_data = int(i * 0.2)
					train_test_str = "train" if train else "test"
					temp_path =  (
						f"{dataset_dir}/L2_{num_samples_L2}Samples_L1_{num_samples}Samples_{num_colored_blocks}_{num_possible_block_pairs}_"
						f"{num_rows}_{num_cols}_{temp_data}_{beta}_{train_test_str}_{utility_mode}/"
					)
					self.dataset_folders.append(temp_path)
				# self.dataset_folders.append(self.dataset_folder)
			else:
				self.dataset_folders = [self.dataset_folder]

	def generate_six_data_points(self):
		rollouts = []
		desire_ints = []

		dataInfo = (self.saved_model_dir, self.device, self.L0_inference_model, self.num_samples, self.num_samples_L2)


		baseEnv = None 
		# with Pool() as pool:
		# 	for (new_rolls, env, desiress) in pool.map(genInitialEnv, [dataInfo]):
		# 		baseEnv = env
		# 		for i, r in enumerate(new_rolls):
		# 			rollout, done = r
		# 			try:
		# 				x = len(done)
		# 				pdb.set_trace()
		# 			except:
		# 				pass
		# 			rollouts.append((rollout, done))
		# 			desire_ints.append(desiress[i])

		new_rolls, env, desiress = default_L1_rollout(saved_model_dir=self.saved_model_dir, device=self.device, inference_model=self.L0_inference_model, num_samples=self.num_samples, num_samples_L2=self.num_samples_L2, useBFS=self.useBFS)
		for i, r in enumerate(new_rolls):
			rollout, done = r
			try:
				x = len(done)
				pdb.set_trace()
			except:
				pass
			rollouts.append((rollout, done))
			desire_ints.append(desiress[i])

		inputArgs = (baseEnv, dataInfo)
		# with Pool() as pool:
		# 	handler = pool.map_async(genAdditionalEnv, [inputArgs, inputArgs])
		# 	for (new_rolls, env, desiress) in handler.get():
		# 		for i, r in enumerate(new_rolls):
		# 			rollout, done = r 
		# 			rollouts.append((rollout, done))
		# 			desire_ints.append(desiress[i])


		for i in range(2):
			colored_block_utilities_0 = envs.construction_sample.sample_block_pair_utilities(
			self.num_possible_block_pairs, return_prob=False
			)
			colored_block_utilities_1 = envs.construction_sample.sample_block_pair_utilities(
			self.num_possible_block_pairs, return_prob=False
			)
			# colored_block_utilities_1 = colored_block_utilities_0
			colored_block_utilities = {
				0: colored_block_utilities_0,
				1: colored_block_utilities_1,
			}
			env.colored_block_utilities = colored_block_utilities
			new_rolls, env, desiress = default_L1_rollout(env=env, saved_model_dir=self.saved_model_dir, device=self.device, inference_model=self.L0_inference_model, num_samples=self.num_samples, num_samples_L2=self.num_samples_L2, useBFS=self.useBFS)
			for i, r in enumerate(new_rolls):
				rollout, done = r 
				rollouts.append((rollout, done))
				desire_ints.append(desiress[i])

		datapoints = []

		# def processRawData(inputArgs):
		# 	i, rolloutTuple = inputArgs
		# 	rollout, done = rolloutTuple
		# 	single_data_point = {}

		# 	try:
		# 		actions_raw = [x[0] for x in rollout]
		# 	except:
		# 		pdb.set_trace()
		# 		actions_raw = [x[0] for x in rollout]
		# 	states_raw = [x[1] for x in rollout]

		# 	num_timesteps = len(states_raw)

		# 	states = torch.stack(
		# 		[
		# 			multi_agent_state_to_state_tensor(state_raw, self.num_colored_blocks, self.device)
		# 			for state_raw in states_raw
		# 		],
		# 		dim=0,
		# 	)
		# 	# Convert actions to tensors
		# 	actions = torch.stack(
		# 		[action_to_action_tensor(action_raw, self.device) for action_raw in actions_raw], dim=0,
		# 	)	
		# 	desire = torch.tensor([desire_ints[i]] * num_timesteps, device=self.device,)

		# 	socialInference = [x[6] for x in rollout]

		# 	#socialInference = [[x[4], x[5]] for x in rollout]

		# 	final_correct = np.argmax(socialInference[-1]) == desire_ints[i]
		# 	num_correct_guesses = np.sum(np.argmax(socialInference, axis=-1) == desire_ints[i])

		# 	single_data_point["final_correct"] = torch.tensor(final_correct, device=self.device)
		# 	single_data_point["num_correct_guesses"] = torch.tensor(num_correct_guesses, device=self.device)
		# 	single_data_point["states"] = states.to(self.device)
		# 	single_data_point["actions"] = actions.to(self.device)
		# 	single_data_point["num_block_pairs"] = torch.tensor(self.num_possible_block_pairs, device=self.device)
		# 	single_data_point["desire"] = desire
		# 	single_data_point["IS_inferences"] = torch.tensor(socialInference, device=self.device)

		# 	datapoints.append(single_data_point)


		# inputArgs = [(i, rollout) for i, rollout in enumerate(rollouts)]
		# with Pool() as pool:
		# 	handler = pool.map_async(processRawData, inputArgs)

		i = 0
		for (rollout, done) in rollouts:
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
					multi_agent_state_to_state_tensor(state_raw, self.num_colored_blocks, self.device)
					for state_raw in states_raw
				],
				dim=0,
			)

			# for state_id, s in enumerate(states):
			# 	actual = states_raw[state_id]
			# 	try:
			# 		converted_back = multi_agent_state_tensor_to_state(s)
			# 		assert converted_back == actual, f"State conversion failed for {actual}"
			# 	except:
			# 		converted_back = multi_agent_state_tensor_to_state(s, L2_actions=L2_actions_raw[state_id - 1])
			# 		pdb.set_trace()
					# multi_agent_state_tensor_to_state(s)

			# Convert actions to tensors
			L1_actions = torch.stack(
				[action_to_action_tensor(action_raw, self.device) for action_raw in L1_actions_raw], dim=0,
			)
			L2_actions = torch.stack(
				[action_to_action_tensor(action_raw, self.device) for action_raw in L2_actions_raw], dim=0,
			)
			desire = torch.tensor([desire_ints[i]] * num_timesteps, device=self.device,)

			socialInference = [x[6] for x in rollout]

			#socialInference = [[x[4], x[5]] for x in rollout]

			final_correct = np.argmax(socialInference[-1]) == desire_ints[i]
			num_correct_guesses = np.sum(np.argmax(socialInference, axis=-1) == desire_ints[i])

			single_data_point["final_correct"] = torch.tensor(final_correct, device=self.device)
			single_data_point["num_correct_guesses"] = torch.tensor(num_correct_guesses, device=self.device)
			single_data_point["states"] = states.to(self.device)
			single_data_point["L1_actions"] = L1_actions.to(self.device)
			single_data_point["L2_actions"] = L2_actions.to(self.device)
			single_data_point["num_block_pairs"] = torch.tensor(self.num_possible_block_pairs, device=self.device)
			single_data_point["desire"] = desire
			single_data_point["IS_inferences"] = torch.tensor(socialInference, device=self.device)

			datapoints.append(single_data_point)

			i += 1

		return datapoints[0], datapoints[1], datapoints[2], datapoints[3], datapoints[4], datapoints[5]

	def generate_six(self, starting):
		self.data = [None] * self.num_data * 6 
		for sample_id in range(self.num_data):
			sId1 = sample_id * 6
			sId2 = sId1 + 1
			sId3 = sId2 + 1 
			sId4 = sId3 + 1
			sId5 = sId4 + 1
			sId6 = sId5 + 1
			path_names = [] 
			for sid in [sId1, sId2, sId3, sId4, sId5, sId6]:  # get file location
				path_name = self.dataset_folder + str(sid) + ".pik"
				path_names.append(path_name)
				self.data[sid] = path_name
			if sample_id >= starting:
				print(f"generating samples {sId1}, {sId2}, {sId3}, {sId4}, {sId5}, {sId6}")

				d1, d2, d3, d4, d5, d6 = self.generate_six_data_points()  # get data
				# while d1 is None:  # getting around timeout instances
				# 	d1, d2, d3, d4, d5, d6 = self.generate_six_data_points()
				with open(path_names[0], "wb") as f:  # save each datapoint to a separate file
					pickle.dump(d1, f)
					f.close()
				with open(path_names[1], "wb") as f:
					pickle.dump(d2, f)
					f.close()
				with open(path_names[2], "wb") as f:
					pickle.dump(d3, f)
					f.close()
				with open(path_names[3], "wb") as f:
					pickle.dump(d4, f)
					f.close()
				with open(path_names[4], "wb") as f:
					pickle.dump(d5, f)
					f.close()
				with open(path_names[5], "wb") as f:
					pickle.dump(d6, f)
					f.close()


	def load(self):

		p = Path(self.dataset_folder)
		if not p.is_dir():
			p.mkdir(parents=True)
		if len(os.listdir(self.dataset_folder)) < self.num_data * 6 and not self.human and not self.synthetic:
			if self.train and self.num_data == 10000 and False:  # don't generate in this case
				pass 
			elif not self.train and self.num_data == 2000 and False:  # don't generate data in this case
				pass
			else:  # generate data in all other cases
				starting = len(os.listdir(self.dataset_folder)) // 6
				self.generate_six(starting)

		# self.data = [None] * self.num_data * 6
		# for sid in range(self.num_data * 6):
		# 	self.data[sid] = self.dataset_folder + str(sid) + ".pik"
		# numInDir = []
		# for folder in self.dataset_folders:
		# 	p = Path(folder)
		# 	if p.is_dir():
		# 		numInDir.append(len(os.listdir(folder)))



		self.data = {}
		i = 0
		currFolder = self.dataset_folders[0]
		# print(currFolder)
		dataRemaining = len(os.listdir(currFolder))
		sid = 0
		# for n in range(self.num_data * 6):
		for n in range(dataRemaining):
			if dataRemaining == 0:
				i += 1
				try:
					currFolder = self.dataset_folders[i]
				except:
					pdb.set_trace()
				dataRemaining = len(os.listdir(currFolder))
				sid = 0
			self.data[n] = currFolder + str(sid) + ".pik"

			sid += 1
			dataRemaining -= 1

		self.num_data = len(self.data)


	def __getitem__(self, idx):
		"""
		Returns
			states: tensor [num_timesteps, num_rows, num_cols, num_colored_blocks + num_colored_blocks + 4]
				states[t, r, c, d] is a one hot where
					d=0 -- nothing
					1 -- wall
					2 -- second agent
					3 to (3 + num_colored_blocks - 1) -- colored block type
					3 + num_colored_blocks -- agent
					3 + num_colored_blocks + 1 to 3 + 2 * num_colored_blocks -- colored_block in inventory
				NOTE: a cell is not represented by a one-hot vector if there is an agent there
			actions: tensor [num_timesteps, num_actions=6] - one hot representation
				the first action is always [0, 0, 0, 0, 0, 0] -- corresponding to no action
				[1, 0, 0, 0, 0, 0] ... [0, 0, 0, 0, 0, 1] correpond to UP, DOWN, LEFT, RIGHT, PUT_DOWN, STOP
				based on envs.construction.Action
			num_block_pairs: tensor representation of self.num_block_pairs
			desire: tensor [] from 0 to (num_possible_colored_blocks! - 1)
		"""
		res = None
		# Load the .pik file with map_location='cpu'
		# pdb.set_trace()
		# data = torch.load(self.data[idx], map_location='cpu')
		with open(self.data[idx], "rb") as f:
		# def _load(obj, *args):
		# 	if isinstance(obj, torch.Tensor) and obj.is_cuda:
		# 		return obj.cpu()
		# 	return obj
		# with open(filename, 'rb') as f:
		# 	data = pickle.load(f, encoding='latin1', fix_imports=True, object_hook=_load)

			data = pickle.load(f)
		try:
			res = (
				data["states"].to(self.device),
				data["actions"].to(self.device),
				data["num_block_pairs"].to(self.device),
				data["desire"].to(self.device),
				data["IS_inferences"].to(self.device),
				data["final_correct"].to(self.device),
				data["num_correct_guesses"].to(self.device)
			)
		except:
			res = (
				data["states"].to(self.device),
				data["L1_actions"].to(self.device),
				data["num_block_pairs"].to(self.device),
				data["desire"].to(self.device),
				data["IS_inferences"].to(self.device),
				data["final_correct"].to(self.device),
				data["num_correct_guesses"].to(self.device),
				data["L2_actions"].to(self.device),
			)
			# f.close()
		return res

	def __len__(self):
		return self.num_data




class ReasoningAboutL0Dataset(torch.utils.data.Dataset):
	def __init__(
		self,
		utility_mode='ranking',
		beta=0.1,
		num_data=1000,
		seed=1,
		dataset_dir=None,
		train=True,
		device="cpu",
		num_rows=10,
		num_cols=10,
		num_colored_blocks=len(envs.construction.ALL_COLORED_BLOCKS),
		num_possible_block_pairs=len(envs.construction.ALL_BLOCK_PAIRS),
		last=1,
		useBFS=False
	):
		self.num_colored_blocks = num_colored_blocks
		self.num_possible_block_pairs = num_possible_block_pairs
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.beta = beta
		self.utility_mode = utility_mode
		self.seed = seed
		self.num_data = num_data
		self.dataset_dir = dataset_dir
		self.train = train
		self.device = device
		self.last = last == 1
		self.useBFS = useBFS

		self.data = {}

		if self.train:
			seed_offset = 0
		else:
			seed_offset = 1
		# random.seed(seed + seed_offset)
		# np.random.seed(seed + seed_offset)
		if dataset_dir:
			train_test_str = "train" if train else "test"
			self.dataset_path = (
				f"{dataset_dir}/{num_colored_blocks}_{num_possible_block_pairs}_"
				f"{num_rows}_{num_cols}_{num_data}_{beta}_0_{train_test_str}_{utility_mode}.pik"
			)

			self.dataset_folder = (
				f"{dataset_dir}/{num_colored_blocks}_{num_possible_block_pairs}_"
				f"{num_rows}_{num_cols}_{num_data}_{beta}_{train_test_str}_{utility_mode}/"
			)
			if num_data == 150000 and train:
				self.dataset_folders = []
				for i in range(11, 21):
					temp_data = i * 1000
					train_test_str = "train" if train else "test"
					temp_path =  (
						f"{dataset_dir}/{num_colored_blocks}_{num_possible_block_pairs}_"
						f"{num_rows}_{num_cols}_{temp_data}_{beta}_{train_test_str}_{utility_mode}/"
					)
					self.dataset_folders.append(temp_path)
				self.dataset_folders.append(self.dataset_folder)
			elif num_data == 30000 and not train:
				self.dataset_folders = []
				for i in range(11, 21):
					temp_data = int(i * 1000 * 0.2)
					train_test_str = "train" if train else "test"
					temp_path =  (
						f"{dataset_dir}/{num_colored_blocks}_{num_possible_block_pairs}_"
						f"{num_rows}_{num_cols}_{temp_data}_{beta}_{train_test_str}_{utility_mode}/"
					)
					self.dataset_folders.append(temp_path)
				# self.dataset_folders.append(self.dataset_folder)
			else:
				self.dataset_folders = [self.dataset_folder]


	def generate_three_data_points(self):
		'''
		Generates all three different utilities for the same initial state
		'''

		# sample a random environment
		env = envs.construction_sample.sample_construction_env(
				num_rows=self.num_rows,
				num_cols=self.num_cols,
				num_colored_blocks=self.num_colored_blocks,
				num_possible_block_pairs=self.num_possible_block_pairs
			)

		different_utils = []
		while len(different_utils) < 3:
			temp_util = envs.construction_sample.sample_block_pair_utilities(self.num_possible_block_pairs)
			if temp_util not in different_utils:
				different_utils.append(temp_util)

		datapoints = []
		# # parallelizing data generation
		# pool = Pool()
		# datapoints = pool.map(self.generate_single_data_point, different_util_environments)
		# datapoints = [d for d in datapoints]
		# pool.close()
		# pool.join()

		for utils in different_utils:
			# pdb.set_trace()
			env_copy = pickle.loads(pickle.dumps(env))
			env_copy.colored_block_utilities = utils

			L0_rollout, done = test_sample_construction_L0_rollout.sample_L0_rollout(env_copy, self.beta, useBFS=self.useBFS)

			if not done and not self.useBFS:  # only for heuristics to assure it reaches terminal state 
				return None, None, None

			single_data_point = {}
			actions_raw = [x[0] for x in L0_rollout]
			states_raw = [x[1] for x in L0_rollout]

			# Convert states to tensors
			num_timesteps = len(states_raw)

			states = torch.stack(
				[
					state_to_state_tensor(state_raw, self.num_colored_blocks, self.device)
					for state_raw in states_raw
				],
				dim=0,
			)
			# Convert actions to tensors
			actions = torch.stack(
				[action_to_action_tensor(action_raw, self.device) for action_raw in actions_raw], dim=0,
			)

			# Extract desire
			if self.utility_mode == "ranking":
				desire_int = block_pair_utilities_to_desire_int(
					env_copy.colored_block_utilities, self.num_possible_block_pairs
				)
			elif self.utility_mode == "top":
				# Which key has the max value in env.block_pair_utilities?
				top_block_pair = max(env_copy.colored_block_utilities, key=env_copy.colored_block_utilities.get)
				desire_int = envs.construction.ALL_BLOCK_PAIRS.index(top_block_pair)

			IS_inferences = get_rollout_gt_inference(states_raw, actions_raw, desire_int, self.num_possible_block_pairs)
			final_correct = np.argmax(IS_inferences[-1]) == desire_int
			num_correct_guesses = np.sum(np.argmax(IS_inferences, axis=-1) == desire_int)

			desire = torch.tensor([desire_int] * num_timesteps, device=self.device,)
			single_data_point["final_correct"] = torch.tensor(final_correct, device=self.device)
			single_data_point["num_correct_guesses"] = torch.tensor(num_correct_guesses, device=self.device)
			single_data_point["states"] = states
			single_data_point["actions"] = actions
			single_data_point["num_block_pairs"] = torch.tensor(self.num_possible_block_pairs, device=self.device)
			single_data_point["desire"] = desire
			single_data_point["IS_inferences"] = torch.tensor(IS_inferences, device=self.device)

			datapoints.append(single_data_point)
		return datapoints[0], datapoints[1], datapoints[2]


	def generate_single_data_point(self, environment=None):
		single_data_point = {}
		if environment is None:
			L0_rollout, done, env = None, False, None
			while not done:
				# Sample a random env
				env = envs.construction_sample.sample_construction_env(
					num_rows=self.num_rows,
					num_cols=self.num_cols,
					num_colored_blocks=self.num_colored_blocks,
					num_possible_block_pairs=self.num_possible_block_pairs
				)

				# Sample a random layout
				L0_rollout, done = test_sample_construction_L0_rollout.sample_L0_rollout(env, self.beta)
		else:
			L0_rollout, done, env = None, False, environment
			while not done:
				env.reset()
				L0_rollout, done = test_sample_construction_L0_rollout.sample_L0_rollout(env, self.beta)

		# Extract a_{1:T}, s_{1:T}
		actions_raw = [x[0] for x in L0_rollout]
		states_raw = [x[1] for x in L0_rollout]

		for i in range(1, len(states_raw)):  # deterministic so we want to make sure just incase
			try:
				assert states_raw[i] == env.transition(states_raw[i-1], actions_raw[i-1])
			except:
				pdb.set_trace()

		# Convert states to tensors
		num_timesteps = len(states_raw)
		states = torch.stack(
			[
				state_to_state_tensor(state_raw, self.num_colored_blocks, self.device)
				for state_raw in states_raw
			],
			dim=0,
		)
		# Convert actions to tensors
		actions = torch.stack(
			[action_to_action_tensor(action_raw, self.device) for action_raw in actions_raw], dim=0,
		)

		# Extract desire
		if self.utility_mode == "ranking":
			desire_int = block_pair_utilities_to_desire_int(
				env.colored_block_utilities, self.num_possible_block_pairs
			)
		elif self.utility_mode == "top":
			# Which key has the max value in env.block_pair_utilities?
			top_block_pair = max(env.colored_block_utilities, key=env.colored_block_utilities.get)
			desire_int = envs.construction.ALL_BLOCK_PAIRS.index(top_block_pair)

		IS_inferences = get_rollout_gt_inference(states_raw, actions_raw, desire_int, self.num_possible_block_pairs) 
		final_correct = np.argmax(IS_inferences[-1]) == desire_int
		num_correct_guesses = np.sum(np.argmax(IS_inferences, axis=-1) == desire_int)
		
		desire = torch.tensor([desire_int] * num_timesteps, device=self.device,)
		
		single_data_point["final_correct"] = torch.tensor(final_correct, device=self.device)
		single_data_point["num_correct_guesses"] = torch.tensor(num_correct_guesses, device=self.device)
		single_data_point["states"] = states
		single_data_point["actions"] = actions
		single_data_point["num_block_pairs"] = torch.tensor(self.num_possible_block_pairs, device=self.device)
		single_data_point["desire"] = desire
		single_data_point["IS_inferences"] = torch.tensor(IS_inferences, device=self.device)
		return single_data_point

	def generate(self):
		self.data = [None] * self.num_data
		for sample_id in range(self.num_data):
			print("generating sample", sample_id)
			self.data[sample_id] = self.generate_single_data_point()

		with open(self.dataset_path, "wb") as f:
			pickle.dump(self.data, f)
			f.close()

	def generate_three(self, starting):
		self.data = [None] * self.num_data * 3 
		for sample_id in range(self.num_data):
			sId1 = sample_id * 3
			sId2 = sId1 + 1
			sId3 = sId1 + 2 
			path_names = [] 
			for sid in [sId1, sId2, sId3]:  # get file location
				path_name = self.dataset_folder + str(sid) + ".pik"
				path_names.append(path_name)
				self.data[sid] = path_name
			if sample_id >= starting:
				print(f"generating samples {sId1}, {sId2}, {sId3}")

				d1, d2, d3 = self.generate_three_data_points()  # get data
				while d1 is None:  # getting around timeout instances
					d1, d2, d3 = self.generate_three_data_points()
				with open(path_names[0], "wb") as f:  # save each datapoint to a separate file
					pickle.dump(d1, f)
					f.close()
				with open(path_names[1], "wb") as f:
					pickle.dump(d2, f)
					f.close()
				with open(path_names[2], "wb") as f:
					pickle.dump(d3, f)
					f.close()
	def load(self):

		p = Path(self.dataset_folder)
		if not p.is_dir():
			p.mkdir(parents=True)
		print(self.dataset_folder)
		if len(os.listdir(self.dataset_folder)) < self.num_data * 3 and (self.num_data != 150000 and self.num_data != 30000):
			starting = len(os.listdir(self.dataset_folder)) // 3
			print(starting)
			self.generate_three(starting)

		self.data = {}
		i = 0
		currFolder = self.dataset_folders[0]
		# print(currFolder)
		dataRemaining = len(os.listdir(currFolder))
		sid = 0
		for n in range(self.num_data * 3):
			if dataRemaining == 0:
				i += 1
				try:
					currFolder = self.dataset_folders[i]
				except:
					pdb.set_trace()
				dataRemaining = len(os.listdir(currFolder))
				sid = 0
			self.data[n] = currFolder + str(sid) + ".pik"

			sid += 1
			dataRemaining -= 1

		self.num_data = len(self.data)


		# with open(self.dataset_path, "rb") as f:
		#     self.data = pickle.load(f)
		#     self.num_data = len(self.data)
		#     f.close()

	def __getitem__(self, idx):
		"""
		Returns
			states: tensor [num_timesteps, num_rows, num_cols, num_colored_blocks + num_colored_blocks + 4]
				states[t, r, c, d] is a one hot where
					d=0 -- nothing
					1 -- wall
					2 -- second agent
					3 to (3 + num_colored_blocks - 1) -- colored block type
					3 + num_colored_blocks -- agent
					3 + num_colored_blocks + 1 to 3 + 2 * num_colored_blocks -- colored_block in inventory
				NOTE: a cell is not represented by a one-hot vector if there is an agent there
			actions: tensor [num_timesteps, num_actions=6] - one hot representation
				the first action is always [0, 0, 0, 0, 0, 0] -- corresponding to no action
				[1, 0, 0, 0, 0, 0] ... [0, 0, 0, 0, 0, 1] correpond to UP, DOWN, LEFT, RIGHT, PUT_DOWN, STOP
				based on envs.construction.Action
			num_block_pairs: tensor representation of self.num_block_pairs
			desire: tensor [] from 0 to (num_possible_colored_blocks! - 1)
		"""
		res = None
		with open(self.data[idx], "rb") as f:
			data = pickle.load(f)
			res = (
				data["states"].to(self.device),
				data["actions"].to(self.device),
				data["num_block_pairs"].to(self.device),
				data["desire"].to(self.device),
				data["IS_inferences"].to(self.device),
				data["final_correct"].to(self.device),
				data["num_correct_guesses"].to(self.device)
			)
			f.close()
		return res

	def __len__(self):
		return self.num_data



def multi_collate(batch):
	lens = [item[0].shape[0] for item in batch]
	idx = np.argsort(lens)
	idx = list(idx[::-1])
	states = [batch[i][0] for i in idx]
	L1_actions = [batch[i][1] for i in idx]
	L2_actions = [batch[i][7] for i in idx]
	desire = torch.cat([batch[i][3] for i in idx], dim=-1)
	IS_inference = []

	numCorrectFinal = 0
	numCorrectOverall = 0

	for i in idx:
		batch_inf = batch[i][4]
		final_correct = batch[i][5].long().cpu().detach().numpy()
		totCorrect = batch[i][6].cpu().detach().numpy()
		numCorrectFinal += final_correct  # does final prediction match gt
		numCorrectOverall += totCorrect
		for inf in batch_inf:
			IS_inference.append(inf.tolist())

	return states, L1_actions, desire, IS_inference, numCorrectFinal / len(idx), numCorrectOverall / len(IS_inference), L2_actions

def multi_collate_last(batch):
	lens = [item[0].shape[0] for item in batch]
	idx = np.argsort(lens)
	idx = list(idx[::-1])
	states = [batch[i][0] for i in idx]
	L1_actions = [batch[i][1] for i in idx]
	L2_actions = [batch[i][7] for i in idx]
	desire = torch.cat([batch[i][3].unsqueeze(0)[:, -1] for i in idx], dim=-1)
	IS_inference = [batch[i][4][-1].tolist() for i in idx]

	numCorrectFinal = 0
	numCorrectOverall = 0

	for i in idx:
		final_correct = batch[i][5].long().cpu().detach().numpy()
		totCorrect = batch[i][6].cpu().detach().numpy()
		numCorrectFinal += final_correct  # does final prediction match gt
		numCorrectOverall += totCorrect

	return states, L1_actions, desire, IS_inference, numCorrectFinal / len(idx), numCorrectOverall / len(IS_inference), L2_actions

def my_collate(batch):
	lens = [item[0].shape[0] for item in batch]
	idx = np.argsort(lens)
	idx = list(idx[::-1])
	states = [batch[i][0] for i in idx]
	actions = [batch[i][1] for i in idx]
	desire = torch.cat([batch[i][3] for i in idx], dim=-1)
	IS_inference = []

	numCorrectFinal = 0
	numCorrectOverall = 0

	for i in idx:
		batch_inf = batch[i][4]
		final_correct = batch[i][5].long().cpu().detach().numpy()
		totCorrect = batch[i][6].cpu().detach().numpy()
		numCorrectFinal += final_correct  # does final prediction match gt
		numCorrectOverall += totCorrect
		for inf in batch_inf:
			IS_inference.append(inf.tolist())

	return states, actions, desire, IS_inference, numCorrectFinal / len(idx), numCorrectOverall / len(IS_inference)


def my_collate_last(batch):
	lens = [item[0].shape[0] for item in batch]
	idx = np.argsort(lens)
	idx = list(idx[::-1])
	states = [batch[i][0] for i in idx]
	actions = [batch[i][1] for i in idx]
	desire = torch.cat([batch[i][3].unsqueeze(0)[:, -1] for i in idx], dim=-1)
	IS_inference = [batch[i][4][-1].tolist() for i in idx]

	numCorrectFinal = 0
	numCorrectOverall = 0

	for i in idx:
		final_correct = batch[i][5].long().cpu().detach().numpy()
		totCorrect = batch[i][6].cpu().detach().numpy()
		numCorrectFinal += final_correct  # does final prediction match gt
		numCorrectOverall += totCorrect

	return states, actions, desire, IS_inference, numCorrectFinal / len(idx), numCorrectOverall / len(IS_inference)


def action_tensor_to_action(action_tensor):
	return envs.construction.Action(utils.general.one_hot_to_int(action_tensor))


def state_tensor_to_state(state_tensor):
	num_rows, num_cols, state_dim = state_tensor.shape
	num_possible_colored_blocks = int((state_dim - 4) / 2)
	cell_values = []
	colored_blocks = {}
	agent_location = None
	block_picked = None
	for row in range(num_rows):
		row_values = ''
		for col in range(num_cols):
			values = utils.general.get_one_ids(state_tensor[row, col])
			for value in values:
				if value == 0:
					row_values += '.'
				elif value == 1:
					row_values += '*'
				elif value == 2:
					row_values += '▲'
				elif value == 3 + num_possible_colored_blocks:
					agent_location = (row, col)
					row_values += '.'
				elif value > 3 + num_possible_colored_blocks:
					agent_location = (row, col)
					block = envs.construction.ALL_COLORED_BLOCKS[value - 3 - num_possible_colored_blocks]
					colored_blocks[block] = (row, col)
					block_picked = block
					row_values += '.'
				else:
					block = envs.construction.ALL_COLORED_BLOCKS[value - 3]
					row_values += block
					colored_blocks[block] = (row, col)
		cell_values.append(row_values)
	if len(colored_blocks) < num_possible_colored_blocks:
		for block in envs.construction.ALL_COLORED_BLOCKS[:num_possible_colored_blocks]:
			if block not in colored_blocks:
				block_picked = block
				colored_blocks[block] = agent_location
	gridworld = envs.construction.Gridworld(cell_values)
	return envs.construction.State(gridworld, agent_location, colored_blocks, block_picked)

def state_to_state_tensor(state, num_colored_blocks, device='cpu', goal_pair=None):
	"""
	Args
		state (envs.construction.State)
	Returns
		state_tensor: tensor [num_rows, num_cols, num_colored_blocks * 2 + 4]
			states[r, c, d] is a one hot where
				d=0 -- nothing
				1 -- wall
				2 -- second agent
				3 to (3 + num_colored_blocks - 1) -- colored block type not in inventory
				3 + num_colored_blocks -- agent
				3 + num_colored_blocks + 1 to 3 + num_colored_blocks * 2 -- colored block type in inventory
			NOTE: a cell is not represented by a one-hot vector if there is an agent there
	"""
	num_rows = state.gridworld.num_rows
	num_cols = state.gridworld.num_cols

	state_tensor = torch.zeros(
		(num_rows, num_cols, num_colored_blocks + 4 + num_colored_blocks,), dtype=torch.long, device=device,
	)

	for row in range(num_rows):
		for col in range(num_cols):
			cell = state.gridworld.map[row][col]
			if cell == '.':
				value = 0
			elif cell == '*':
				value = 1
			elif cell == '▲':
				value = 2
			elif cell == '●':
				value = 3 + num_colored_blocks
			else:
				colored_block_id = envs.construction.ALL_COLORED_BLOCKS.index(cell)
				value = 3 + colored_block_id

			if (row, col) == state.agent_location:
				value = 3 + num_colored_blocks
				if state.block_picked is not None:
					value += envs.construction.ALL_COLORED_BLOCKS.index(state.block_picked)
			state_tensor[row, col, value] = 1

			if goal_pair is not None and (cell == goal_pair[0] or cell == goal_pair[1]):  # encoding the goal into the state representation
				state_tensor[row, col, value] += 1

	return state_tensor



def multi_agent_state_to_state_tensor(state, num_colored_blocks, device='cpu'):
	"""
	Args
		state (envs.construction.StateMultiAgent)
	Returns
		state_tensor: tensor [num_rows, num_cols, num_colored_blocks * 2 + 4]
			states[r, c, d] is a one hot where
				d=0 -- nothing
				1 -- wall
				2 -- second agent
				3 to (3 + num_colored_blocks - 1) -- colored block type not in inventory
				3 + num_colored_blocks -- agent
				3 + num_colored_blocks + 1 to 3 + num_colored_blocks * 2 -- colored block type in inventory of agent 1
				4 + num_colored_blocks * 2 to 3 + num_colored_blocks * 3 -- colored block type in inventory of agent 2
			NOTE: a cell is not represented by a one-hot vector if there is an agent there
	"""

	num_rows = state.gridworld.num_rows
	num_cols = state.gridworld.num_cols

	state_tensor = torch.zeros(
		(num_rows, num_cols, 3 * num_colored_blocks + 4), dtype=torch.long, device=device,
	)

	for row in range(num_rows):
		for col in range(num_cols):
			cell = state.gridworld.map[row][col]
			if cell == '.':
				value = 0
			elif cell == '*':
				value = 1
			elif cell == '▲':
				value = 2
			elif cell == '●':
				value = 3 + num_colored_blocks
			else:
				colored_block_id = envs.construction.ALL_COLORED_BLOCKS.index(cell)
				value = 3 + colored_block_id

			if (row, col) == state.agent_locations[0]:
				value = 3 + num_colored_blocks
				if state.agent_inv[0] is not None:
					value += envs.construction.ALL_COLORED_BLOCKS.index(state.agent_inv[0]) + 1
			if (row, col) == state.agent_locations[1]:
				value = 2
				if state.agent_inv[1] is not None:
					value = 3 + num_colored_blocks * 2
					value += envs.construction.ALL_COLORED_BLOCKS.index(state.agent_inv[1]) + 1


			state_tensor[row, col, value] = 1

			# if goal_pair is not None and (cell == goal_pair[0] or cell == goal_pair[1]):  # encoding the goal into the state representation
			# 	state_tensor[row, col, value] += 1

	return state_tensor

def multi_agent_state_tensor_to_state(state_tensor, num_colored_blocks=10, device='cpu', L2_actions=None, getL1=False):
	num_rows, num_cols, state_dim = state_tensor.shape
	num_possible_colored_blocks = int((state_dim - 4) / 3)
	cell_values = []
	colored_blocks = {}
	agent_locations = {}
	agent_inv = {0: None, 1:None}
	foundEqual = False
	for row in range(num_rows):
		row_values = ''
		for col in range(num_cols):
			values = utils.general.get_one_ids(state_tensor[row, col])
			for value in values:
				if value == 0:  # found empty
					row_values += '.'
				elif value == 1:  # found a wall
					row_values += '*'
				elif value == 2:  # found agent 1
					row_values += '.'
					agent_locations[1] = (row, col)
				elif value == 3 + num_possible_colored_blocks:  # found agent 0 
					agent_locations[0] = (row, col)
					row_values += '.'
				elif value > 3 + num_possible_colored_blocks and value < 4 + num_possible_colored_blocks * 2:  # agent 0 had inventory
					agent_locations[0] = (row, col)
					block = envs.construction.ALL_COLORED_BLOCKS[value - 4 - num_possible_colored_blocks]
					colored_blocks[block] = (row, col)
					agent_inv[0] = block
					row_values += '.'
				elif value >= 4 + num_possible_colored_blocks * 2:  # agent 1 had inventory 
					agent_locations[1] = (row, col)
					block = envs.construction.ALL_COLORED_BLOCKS[value - 4 - (num_possible_colored_blocks * 2)]
					colored_blocks[block] = (row, col)
					agent_inv[1] = block
					row_values += '.'
				else:
					block = envs.construction.ALL_COLORED_BLOCKS[value - 3]
					if block == "=":
						foundEqual = True
					row_values += block
					colored_blocks[block] = (row, col)
		cell_values.append(row_values)
	if len(colored_blocks) < num_possible_colored_blocks:
		assert L2_actions is not None, "Need previous L2 action that led to this state to proceed"
		for block in envs.construction.ALL_COLORED_BLOCKS[:num_possible_colored_blocks]:
			if block not in colored_blocks:
				if envs.construction.Action(L2_actions.value) == envs.construction.Action.PUT_DOWN:  # if L2 put the block down previously, then the missing block is where L2 is
					colored_blocks[block] = agent_locations[0]
				else:
					colored_blocks[block] = agent_locations[1]

				updatedMap = ""
				for t in range(len(cell_values[colored_blocks[block][0]])):
					if t == colored_blocks[block][1]:
						updatedMap += block 
					else:
						updatedMap += cell_values[colored_blocks[block][0]][t]
				cell_values[colored_blocks[block][0]] = updatedMap

	gridworld = envs.construction.Gridworld(cell_values)
	if getL1:
		L0_state = envs.construction.State(gridworld, agent_locations[0], colored_blocks, agent_inv[0])
		L1_state = envs.construction.StateL1(envs.construction_sample.sample_block_pair_utilities(45), (L0_state, 1.0), L0_state, agent_locations[1], agent_inv[1], prev_action_0=L2_actions)
		return L1_state
	else:
		multi_state =  envs.construction.StateMultiAgent(gridworld, agent_locations, colored_blocks, agent_inv)
		# if foundEqual:
		# 	pdb.set_trace()
		return multi_state


def action_to_action_tensor(action, device):
	"""
	Args
		action (envs.food_trucks.Action)
	Returns
		action_tensor: tensor [num_actions=5]
			the first action is always [0, 0, 0, 0, 0] -- corresponding to no action
			[1, 0, 0, 0, 0] ... [0, 0, 0, 0, 1] correpond to UP, DOWN, LEFT, RIGHT, PUT_DOWN
			based on envs.construction.Action
	"""
	action_tensor = torch.zeros(
		(len(list(envs.construction.Action)),), dtype=torch.long, device=device,
	)
	if action is not None:
		action_tensor[action.value] = 1

	return action_tensor


def desire_int_to_utilities(desire_int, num_possible_block_pairs):
	# utilities_permutations = list(
	#     itertools.permutations(envs.construction.ALL_UTILITIES[:num_possible_block_pairs])
	# )
	# final_utils_perm = []
	# for util in utilities_permutations:
	#     if util not in final_utils_perm:
	#         final_utils_perm.append(util)
	# utilities = final_utils_perm[desire_int]
	utilities = [0] * num_possible_block_pairs
	utilities[desire_int] = 100
	return dict(zip(envs.construction.ALL_BLOCK_PAIRS[:num_possible_block_pairs], utilities))


def block_pair_utilities_to_desire_int(colored_block_utilities, num_possible_block_pairs=3):
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
		utilities.index(100)
	except:
		pdb.set_trace()
	return utilities.index(100)


def save_rollout(gif_path, states, actions, num_possible_block_pairs, desires):
	tmp_dir = utils.general.get_tmp_dir()
	img_paths = []
	for timestep in range(len(states)):
		state = states[timestep]
		action = actions[timestep]

		fig, ax = plt.subplots(1, 1)
		state_tensor_to_state(state).plot(ax=ax)
		ax.set_title(
			"Utilities = "
			+ ", ".join(
				[
					f"{k}: {v}"
					for k, v in desire_int_to_utilities(
						desires[0].item(), num_possible_block_pairs
					).items()
				]
			)
			+ f"\nAction = {action_tensor_to_action(action)}"
		)
		img_path = f"{tmp_dir}/{timestep}.png"
		utils.general.save_fig(fig, img_path)
		img_paths.append(img_path)

	utils.general.make_gif(img_paths, gif_path, 3)
	shutil.rmtree(tmp_dir)
	print("Done")


def plot_dataset(dataset):
	for data_id in range(len(dataset)):
		states, actions, num_block_pairs, desires = dataset[data_id]
		save_rollout(f"{dataset.dataset_path[:-4]}/{data_id}.gif", states, actions, num_block_pairs.item(), desires)


def get_rollout_gt_inference(states_raw, actions_raw, desire, num_possible_block_pairs, num_samples=45):
	IS_inferences = []
	rollout_desire_int = desire
	initial_state = states_raw[0].clone()
	colored_block_desire = desire_int_to_utilities(int(rollout_desire_int), num_possible_block_pairs)
	rollout_env = envs.construction.ConstructionEnv(initial_state, colored_block_desire)

	all_inferences = test_reasoning_about_construction_L0.particle_inference(rollout_env, states_raw, actions_raw, 
		num_samples= num_samples, resample=False, rejuvenate=False, output_every_timestep=True)

	for j in range(len(all_inferences)):
		posterior_belief = test_reasoning_about_construction_L0.get_posterior(all_inferences[j][0], all_inferences[j][1], sort_posterior=False)
		
		posterior_distrib = [None] * len(posterior_belief)
		for p in posterior_belief:  # doing this to preserve order in predictions
			inferred_util_idx = block_pair_utilities_to_desire_int(dict(p[0]), num_possible_block_pairs)  # going from utility to index for consistency
			inferred_util_pred = p[1]  # what is the actual probability assigned to this belief
			posterior_distrib[inferred_util_idx] = inferred_util_pred
		IS_inferences.append(posterior_distrib)

	# try:
	#     assert np.argmax(IS_inferences[-1]) == desire  # did we get the correct prediction on the final inference
	# except:
	#     pdb.set_trace()

	return IS_inferences





if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn')
	sample = envs.construction_sample.sample_multi_agent_env()
	s_tensor = multi_agent_state_to_state_tensor(sample.state)
	converted_s = multi_agent_state_tensor_to_state(s_tensor)
	print(converted_s)


	# data_creator = ReasoningAboutL0Dataset()
	# # for i in range(1):
	# # datapoint = data_creator.generate_single_data_point()
	# d1, d2, d3 = data_creator.generate_three_data_points()
	# for s in d1['states']:
	# 	converted_state = state_tensor_to_state(s)
	# 	print(converted_state)
