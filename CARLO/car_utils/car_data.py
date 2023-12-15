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
import test_scenario1
import test_scenario2
import reasoning_about_car_L0
import reasoning_about_car_L1
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point, Line
from scenario import Scenario1, Action, get_partial_states
import matplotlib.pyplot as plt
import shutil
from torch.multiprocessing import set_start_method, Pool, Process
from scenario import Scenario1, Action


# FORWARD = (0, 5000)
# LEFT = (0.42, -50)
# RIGHT = (-0.9, -170)
# STOP = (0, -1e7)
# SIGNAL = (0, 0)

FORWARD = (0, 4000)
LEFT = (0.5, 0)
RIGHT = (-1.15, -1100)
STOP = (0, -1e7)
SIGNAL = (0, 0)

actionStringDict = {FORWARD:"forward", LEFT:"left", RIGHT:"right", STOP:"stop", SIGNAL:"signal"}
actionIntDict = {FORWARD:0, LEFT:1, RIGHT:2, STOP:3, SIGNAL:4}


class StateBeliefDataset(torch.utils.data.Dataset):
	def __init__(
		self,
		beta=0.1,
		num_data=1000,
		seed=1,
		dataset_dir=None,
		train=True,
		device="cpu",
		num_inference_samples=3,
		sampled_actions = 1,
		lookAheadDepth = 1,
		car1_exist_prior=0.5,
		car2_exist_prior=0.5
	):
		self.beta = beta
		self.seed = seed
		self.num_data = num_data
		self.dataset_dir = dataset_dir
		self.train = train
		self.device = device
		self.num_inference_samples = num_inference_samples
		self.sampled_actions = sampled_actions
		self.lookAheadDepth = lookAheadDepth
		self.car1_exist_prior = car1_exist_prior
		self.car2_exist_prior = car2_exist_prior

		self.data = {}

		if self.dataset_dir:
			train_test_str = "train" if self.train else "test"
			
			self.dataset_folder = (
				f"{dataset_dir}/stateBelief_3car_{num_data}dat_{sampled_actions}_{lookAheadDepth}_{car1_exist_prior}_{car2_exist_prior}_{beta}_{train_test_str}/"
			)

			if self.num_data == 1000 and self.train:
				self.dataset_folders = []
				for i in range(100, 300, 10):
					temp_data = i
					temp_path = (
						f"{dataset_dir}/stateBelief_3car_{temp_data}dat_{sampled_actions}_{lookAheadDepth}_{car1_exist_prior}_{car2_exist_prior}_{beta}_{train_test_str}/"
					)
					self.dataset_folders.append(temp_path)
			elif self.num_data == 100 and not self.train:
				self.dataset_folders = []
				for i in range(10, 30):
					temp_data = i
					temp_path = (
						f"{dataset_dir}/stateBelief_3car_{temp_data}dat_{sampled_actions}_{lookAheadDepth}_{car1_exist_prior}_{car2_exist_prior}_{beta}_{train_test_str}/"
					)
					self.dataset_folders.append(temp_path)
			else:
				self.dataset_folders = [self.dataset_folder]

	def generate_single_scenario_data(self):
		
		# iterate through other agents
		# only do particle inference from states in which other agents are visible
		# fill in missing probabilities with uniform likelihood
		# save complete state action tensor though for every timestep


		rollout = test_scenario1.generate_scenario1_rollout(self.car1_exist_prior, None, None, self.sampled_actions, self.lookAheadDepth)
		scenario, full_observed_state_history, action_history, ground_truth_goals, done, belief_tensor_history, full_action_history = rollout

		num_timesteps = len(full_observed_state_history)
		num_possible_actions = 5
		num_possible_goals = 3

		datapoints = []  # I think this makes 3 datapoints

		# get partial states for each agent
		partial_observations = {}
		for agent_id in action_history:
			partial_observation_trajectory = []
			for full_state in full_observed_state_history:
				partial_obs = get_partial_states(full_state, id=agent_id)
				partial_observation_trajectory.append(partial_obs)
			partial_observations[agent_id] = partial_observation_trajectory

		for agent_id in partial_observations:
			single_data_point = {}

			partial_observation_trajectory = partial_observations[agent_id]


			def get_partial_actions(time):
				partial_actions = pickle.loads(pickle.dumps(action_history))
				for a in partial_actions:  # get actions for every agent at time i
					partial_actions[a] = partial_actions[a][time]
				return partial_actions

			state_action_tensors = torch.stack(
				[
					state_action_to_joint_tensor(state_raw, get_partial_actions(i), self.device)
					for i, state_raw in enumerate(partial_observation_trajectory)
				],
				dim=0,
			)


			existTensors = []
			telemetryTensors = []
			for belief_tensor in belief_tensor_history[agent_id]:
				# just select the first element as the probability
				existTensor = []
				telemetryTensor = []
				for row in belief_tensor:
					existTensor.append([1-row[0], row[0]])  # first item is a probability of not existing
					telemetryTensor.append([row[1], row[2], row[3]]) # second item is the telemetry data
				existTensors.append(existTensor)
				telemetryTensors.append(telemetryTensor)



			single_data_point['state_actions'] = state_action_tensors
			single_data_point['exist_tensors'] = torch.tensor(existTensors)
			single_data_point['telemetry_tensors'] = torch.tensor(telemetryTensors)
			gt_agent_tensor = torch.zeros(16)
			for aID in partial_observations.keys():
				gt_agent_tensor[aID] = 1

			single_data_point['gt_agents'] = gt_agent_tensor
			datapoints.append(single_data_point)

		return datapoints

	def generate_three(self, starting):
		self.data = [None] * self.num_data * 3
		for sample_id in range(self.num_data):
			sId1 = sample_id * 3
			sId2 = sId1 + 1
			path_names = [] 
			for sid in range(sId1, sId1+3):  # get file location
				path_name = self.dataset_folder + str(sid) + ".pik"
				path_names.append(path_name)
				self.data[sid] = path_name
			if sample_id >= starting:
				print(f"generating samples {sId1} through {sId1 + 2} inclusive")

				datapoints = self.generate_single_scenario_data()  # get data
				assert len(datapoints) >= 2

				for i, name in enumerate(path_names):
					with open(name, "wb") as f:# save each datapoint to a separate file
						pickle.dump(datapoints[i], f)
						f.close()

	def load(self):
		p = Path(self.dataset_folder)
		if not p.is_dir():
			p.mkdir(parents=True)
		print(self.dataset_folder)
		if len(os.listdir(self.dataset_folder)) < self.num_data * 3 and \
		((self.num_data != 1000 and self.train) or (self.num_data != 100 and not self.train)):
			starting = len(os.listdir(self.dataset_folder)) // 3
			self.generate_three(starting)

		self.data = {}
		i = 0
		currFolder = self.dataset_folders[i]
		# print(currFolder)
		dataRemaining = len(os.listdir(currFolder))
		sid = 0
		for n in range(self.num_data * 3):
			if dataRemaining == 0:
				i += 1
				try:
					currFolder = self.dataset_folders[i]
				except:
					break
				dataRemaining = len(os.listdir(currFolder))
				sid = 0
			self.data[n] = currFolder + str(sid) + ".pik"

			sid += 1
			dataRemaining -= 1

		self.num_data = len(self.data)

	def __getitem__(self, idx):
		'''
		returns tuple of 
		'''
		res = None
		with open(self.data[idx], "rb") as f:
			data = pickle.load(f)
			res = (
				data["state_actions"].to(self.device),
				data["exist_tensors"].to(self.device),
				data["telemetry_tensors"].to(self.device),
				data['gt_agents'].to(self.device)
			)
			f.close()
		return res

	def __len__(self):
		return self.num_data



class ReasoningAboutScenario1L0Dataset(torch.utils.data.Dataset):
	def __init__(
		self,
		beta=0.1,
		num_data=1000,
		seed=1,
		dataset_dir=None,
		train=True,
		device="cpu",
		num_inference_samples=3,
		sampled_actions = 1,
		lookAheadDepth = 1,
		car_exist_prior=0.65,
		state_belief_model=None,
		exist_belief_model=None
	):
		self.beta = beta
		self.seed = seed
		self.num_data = num_data
		self.dataset_dir = dataset_dir
		self.train = train
		self.device = device
		self.num_inference_samples = num_inference_samples
		self.sampled_actions = sampled_actions
		self.lookAheadDepth = lookAheadDepth
		self.car_exist_prior = car_exist_prior

		self.state_belief_model = state_belief_model
		self.exist_belief_model = exist_belief_model

		self.data = {}

		if self.dataset_dir:
			train_test_str = "train" if self.train else "test"
			
			self.dataset_folder = (
				f"{dataset_dir}/scenario1_3car_{num_data}dat_{sampled_actions}_{lookAheadDepth}_{car_exist_prior}_{beta}_{train_test_str}/"
			)

			if self.num_data == 1000 and self.train:
				self.dataset_folders = []
				for i in range(10, 30):
					temp_data = i * 10
					temp_path = (
						f"{dataset_dir}/scenario1_3car_{temp_data}dat_{sampled_actions}_{lookAheadDepth}_{car_exist_prior}_{beta}_{train_test_str}/"
					)
					self.dataset_folders.append(temp_path)
			elif self.num_data == 100 and not self.train:
				self.dataset_folders = []
				for i in range(10, 30):
					temp_data = i
					temp_path = (
						f"{dataset_dir}/scenario1_3car_{temp_data}dat_{sampled_actions}_{lookAheadDepth}_{car_exist_prior}_{beta}_{train_test_str}/"
					)
					self.dataset_folders.append(temp_path)
			else:
				self.dataset_folders = [self.dataset_folder]

	def generate_single_scenario_data(self):
		
		# iterate through other agents
		# only do particle inference from states in which other agents are visible
		# fill in missing probabilities with uniform likelihood
		# save complete state action tensor though for every timestep


		rollout = test_scenario1.generate_scenario1_rollout(self.car_exist_prior, self.state_belief_model, self.exist_belief_model, self.sampled_actions, self.lookAheadDepth)
		scenario, full_observed_state_history, action_history, ground_truth_goals, done, belief_tensor_history, full_action_history = rollout

		num_timesteps = len(full_observed_state_history)
		num_possible_actions = 5
		num_possible_goals = 3

		datapoints = []  # I think this makes at most 6 datapoints

		# get partial states for each agent
		partial_observations = {}
		for agent_id in action_history:
			partial_observation_trajectory = []
			for full_state in full_observed_state_history:
				partial_obs = get_partial_states(full_state, id=agent_id)
				partial_observation_trajectory.append(partial_obs)
			partial_observations[agent_id] = partial_observation_trajectory


		for agent_id in partial_observations:
			partial_observation_trajectory = partial_observations[agent_id]

			def get_partial_actions(time):
				partial_actions = pickle.loads(pickle.dumps(action_history))
				for a in partial_actions:  # get actions for every agent at time i
					partial_actions[a] = partial_actions[a][time]
				return partial_actions

			state_action_tensors = torch.stack(
				[
					state_action_to_joint_tensor(state_raw, get_partial_actions(i), self.device)
					for i, state_raw in enumerate(partial_observation_trajectory)
				],
				dim=0,
			)


			for other_agent_id in action_history:
				if other_agent_id != agent_id:
					single_data_point = {}

					id_tensors = agent_pair_to_tensor(agent_id, other_agent_id, self.device)

					other_goal_idx = reasoning_about_car_L0.GOAL_SPACE.index(ground_truth_goals[other_agent_id])
					other_desire = torch.tensor([other_goal_idx] * num_timesteps, device=self.device,)


					IS_goal_inference = []
					IS_action_inference = []

					foundAgent = False
					startTime = None
					endTime = -1
					for timestep, obs in enumerate(partial_observation_trajectory):
						agentExists = obs.agent_exists(other_agent_id)

						if not agentExists: # if we don't see the agent, give uniform probs
							IS_goal_inference.append([1/num_possible_goals] * num_possible_goals)
							IS_action_inference.append([1/num_possible_actions] * num_possible_actions)
						else:  # once we've found the agent, flag it down and do inference from there
							foundAgent = True
							startTime = timestep
							break
					if foundAgent:
						# pdb.set_trace()
						goalInference, actionInference = reasoning_about_car_L0.get_car_L0_is_inference(
							scenario, 
							partial_observation_trajectory[startTime:], 
							action_history[other_agent_id][startTime:], 
							other_agent_id, 
							num_samples=self.num_inference_samples, 
							sampled_actions=self.sampled_actions, 
							lookAheadDepth=self.lookAheadDepth,
							full_action_history=full_action_history[startTime:],
							state_belief_model=self.state_belief_model,
							exist_belief_model=self.exist_belief_model
						)
						IS_goal_inference += goalInference.tolist()
						IS_action_inference += actionInference.tolist()

					try:
						assert len(IS_goal_inference) == len(state_action_tensors)
						assert len(state_action_tensors) > 2
					except:
						pdb.set_trace()

					single_data_point["state_actions"] = state_action_tensors
					single_data_point["id_pair"] = id_tensors
					single_data_point["IS_goal_inference"] = torch.tensor(IS_goal_inference, device=self.device)
					single_data_point["IS_action_inference"] = torch.tensor(IS_action_inference, device=self.device)
					single_data_point["other_agent_true_goal"] = other_desire

					gt_agent_tensor = torch.zeros(16)
					for aID in partial_observations.keys():
						gt_agent_tensor[aID] = 1

					single_data_point['gt_agents'] = gt_agent_tensor
					datapoints.append(single_data_point)
		return datapoints

	def generate_two(self, starting):
		self.data = [None] * self.num_data * 6
		for sample_id in range(self.num_data):
			sId1 = sample_id * 6
			sId2 = sId1 + 1
			path_names = [] 
			for sid in range(sId1, sId1+6):  # get file location
				path_name = self.dataset_folder + str(sid) + ".pik"
				path_names.append(path_name)
				self.data[sid] = path_name
			if sample_id >= starting:
				print(f"generating samples {sId1} through {sId1 + 5} inclusive")

				datapoints = self.generate_single_scenario_data()  # get data
				assert len(datapoints) >= 2

				for i, name in enumerate(path_names):
					with open(name, "wb") as f:# save each datapoint to a separate file
						pickle.dump(datapoints[i], f)
						f.close()

	def load(self):
		p = Path(self.dataset_folder)
		if not p.is_dir():
			p.mkdir(parents=True)
		print(self.dataset_folder)
		if len(os.listdir(self.dataset_folder)) < self.num_data * 6 and \
		((self.num_data != 1000 and self.train) or (self.num_data != 100 and not self.train)):
			starting = len(os.listdir(self.dataset_folder)) // 6
			self.generate_two(starting)

		self.data = {}
		i = 0
		currFolder = self.dataset_folders[i]
		# print(currFolder)
		dataRemaining = len(os.listdir(currFolder))
		sid = 0
		tot = self.num_data * 6
		if self.num_data == 100 and not self.train:
			tot = self.num_data
		for n in range(tot):
			if dataRemaining == 0:
				i += 1
				try:
					currFolder = self.dataset_folders[i]
				except:
					break
				dataRemaining = len(os.listdir(currFolder))
				sid = 0
			self.data[n] = currFolder + str(sid) + ".pik"

			sid += 1
			dataRemaining -= 1

		self.num_data = len(self.data)

	def __getitem__(self, idx):
		'''
		returns tuple of 
		'''
		res = None
		with open(self.data[idx], "rb") as f:
			data = pickle.load(f)
			res = (
				data["state_actions"].to(self.device),
				data["id_pair"].to(self.device),
				data["IS_goal_inference"].to(self.device),
				data["IS_action_inference"].to(self.device),
				data["other_agent_true_goal"].to(self.device),
				data['gt_agents'].to(self.device)
			)
			f.close()
		return res

	def __len__(self):
		return self.num_data



class ReasoningAboutScenario2L1Dataset(torch.utils.data.Dataset):
	def __init__(
		self,
		beta=0.1,
		num_data=1000,
		seed=1,
		dataset_dir=None,
		train=True,
		device="cpu",
		num_inference_samples=3,
		sampled_actions = 1,
		lookAheadDepth = 1,
		car1_exist_prior=0.5,
		car2_exist_prior=0.5,
		L0_inference_model=None,
		other_agent_inference_algorithm="Online_IS+NN",
		other_agent_num_samples=1,
		state_model=None,
		exist_model=None,
	):
		self.beta = beta
		self.seed = seed
		self.num_data = num_data
		self.dataset_dir = dataset_dir
		self.train = train
		self.device = device
		self.num_inference_samples = num_inference_samples
		self.sampled_actions = sampled_actions
		self.lookAheadDepth = lookAheadDepth
		self.car1_exist_prior = car1_exist_prior
		self.car2_exist_prior = car2_exist_prior
		self.L0_inference_model = L0_inference_model
		self.other_agent_inference_algorithm = other_agent_inference_algorithm
		self.other_agent_num_samples = other_agent_num_samples
		self.state_model = state_model
		self.exist_model = exist_model

		self.data = {}

		if self.dataset_dir:
			train_test_str = "train" if self.train else "test"
			
			self.dataset_folder = (
				f"{dataset_dir}/scenario2_3car_{num_data}dat_{sampled_actions}_{lookAheadDepth}_{car1_exist_prior}_{car2_exist_prior}_{beta}_{train_test_str}/"
			)

			if self.num_data == 1000 and self.train:
				self.dataset_folders = []
				for i in range(10, 30):
					temp_data = i * 10
					temp_path = (
						f"{dataset_dir}/scenario2_3car_{temp_data}dat_{sampled_actions}_{lookAheadDepth}_{car1_exist_prior}_{car2_exist_prior}_{beta}_{train_test_str}/"
					)
					self.dataset_folders.append(temp_path)
			elif self.num_data == 1000 and not self.train:
				self.dataset_folders = []
				for i in range(10, 30):
					temp_data = i * 10
					temp_path = (
						f"{dataset_dir}/scenario2_3car_{temp_data}dat_{sampled_actions}_{lookAheadDepth}_{car1_exist_prior}_{car2_exist_prior}_{beta}_{train_test_str}/"
					)
					self.dataset_folders.append(temp_path)
			else:
				self.dataset_folders = [self.dataset_folder]

	def generate_single_scenario_data(self):
		
		# iterate through other agents
		# only do particle inference from states in which other agents are visible
		# fill in missing probabilities with uniform likelihood
		# save complete state action tensor though for every timestep


		rollout = test_scenario2.generate_scenario2_rollout(self.car1_exist_prior, self.car2_exist_prior, self.sampled_actions, self.lookAheadDepth,
			num_samples=self.other_agent_num_samples, other_agent_inference_algorithm=self.other_agent_inference_algorithm,
			other_agent_inference_model=self.L0_inference_model, state_model=self.state_model, exist_model=self.exist_model)
		scenario, full_observed_state_history, action_history, ground_truth_goals, done, action_probs = rollout

		num_timesteps = len(full_observed_state_history)
		num_possible_actions = 5
		num_possible_goals = 3

		datapoints = []  # I think this makes at most 6 datapoints

		# get partial states for each agent
		partial_observations = {}
		for agent_id in action_history:
			partial_observation_trajectory = []
			for full_state in full_observed_state_history:
				try:
					partial_obs = get_partial_states(full_state[0], id=agent_id)
				except:
					pdb.set_trace()
				partial_observation_trajectory.append((partial_obs, full_state[1]))
			partial_observations[agent_id] = partial_observation_trajectory


		for agent_id in partial_observations:
			partial_observation_trajectory = partial_observations[agent_id]

			for other_agent_id in action_history:
				if other_agent_id != agent_id:
					single_data_point = {}

					
					def get_partial_actions(time):
						# generate tensors for state and id pair
						partial_actions = pickle.loads(pickle.dumps(action_history))
						for a in partial_actions:  # get actions for every agent at time i
							partial_actions[a] = partial_actions[a][time]
						return partial_actions

					state_action_tensors = torch.stack(
						[
							state_action_to_joint_tensor(state_raw[0], get_partial_actions(i), self.device)
							for i, state_raw in enumerate(partial_observation_trajectory)
						],
						dim=0,
					)

					id_tensors = agent_pair_to_tensor(agent_id, other_agent_id, self.device)

					other_goal_idx = reasoning_about_car_L0.GOAL_SPACE.index(ground_truth_goals[other_agent_id])
					other_desire = torch.tensor([other_goal_idx] * num_timesteps, device=self.device,)


					IS_goal_inference = []
					IS_action_inference = []

					foundAgent = False
					startTime = None
					endTime = -1
					for timestep, obs in enumerate(partial_observation_trajectory):
						agentExists = obs[0].agent_exists(other_agent_id)

						if not agentExists: # if we don't see the agent, give uniform probs
							IS_goal_inference.append([1/num_possible_goals] * num_possible_goals)
							IS_action_inference.append([1/num_possible_actions] * num_possible_actions)
						else:  # once we've found the agent, flag it down and do inference from there
							foundAgent = True
							startTime = timestep
							break
					if foundAgent:
						# pdb.set_trace()
						goalInference, actionInference = reasoning_about_car_L1.get_car_L1_is_inference(
							scenario, 
							partial_observation_trajectory[startTime:], 
							action_history[other_agent_id][startTime:], 
							other_agent_id, 
							num_samples=self.num_inference_samples, 
							sampled_actions=self.sampled_actions, 
							lookAheadDepth=self.lookAheadDepth,
							L0_inference_model=self.L0_inference_model,
							signal_danger_prior=0.5,
							carExistPrior=self.car1_exist_prior,
							other_agent_inference_algorithm=self.other_agent_inference_algorithm,
							other_agent_num_samples=self.other_agent_num_samples,
							state_model=self.state_model,
							exist_model=self.exist_model
						)
						IS_goal_inference += goalInference.tolist()
						IS_action_inference += actionInference.tolist()

					try:
						assert len(IS_goal_inference) == len(state_action_tensors)
						assert len(state_action_tensors) > 2
					except:
						pdb.set_trace()

					single_data_point["state_actions"] = state_action_tensors
					single_data_point["id_pair"] = id_tensors
					single_data_point["IS_goal_inference"] = torch.tensor(IS_goal_inference, device=self.device)
					single_data_point["IS_action_inference"] = torch.tensor(IS_action_inference, device=self.device)
					single_data_point["other_agent_true_goal"] = other_desire
					gt_agent_tensor = torch.zeros(16)
					for aID in partial_observations.keys():
						gt_agent_tensor[aID] = 1

					single_data_point['gt_agents'] = gt_agent_tensor
					datapoints.append(single_data_point)
		return datapoints

	def generate_two(self, starting):
		self.data = [None] * self.num_data * 6
		for sample_id in range(self.num_data):
			sId1 = sample_id * 6
			sId2 = sId1 + 1
			path_names = [] 
			for sid in range(sId1, sId1+6):  # get file location
				path_name = self.dataset_folder + str(sid) + ".pik"
				path_names.append(path_name)
				self.data[sid] = path_name
			if sample_id >= starting:
				print(f"generating samples {sId1} through {sId1 + 5} inclusive")

				datapoints = self.generate_single_scenario_data()  # get data
				assert len(datapoints) >= 2

				for i, name in enumerate(path_names):
					with open(name, "wb") as f:# save each datapoint to a separate file
						pickle.dump(datapoints[i], f)
						f.close()

	def load(self):
		p = Path(self.dataset_folder)
		if not p.is_dir():
			p.mkdir(parents=True)
		print(self.dataset_folder)
		if len(os.listdir(self.dataset_folder)) > 100:
			pass 
		else:
			if len(os.listdir(self.dataset_folder)) < self.num_data * 6 and \
			((self.num_data != 1000 and self.train) or (self.num_data != 1000 and not self.train)):
				starting = len(os.listdir(self.dataset_folder)) // 6
				self.generate_two(starting)

		self.data = {}
		i = 0
		currFolder = self.dataset_folders[i]
		# print(currFolder)
		dataRemaining = len(os.listdir(currFolder))
		while (dataRemaining == 0):
			i += 1
			currFolder = self.dataset_folders[i]
			try:
				dataRemaining = len(os.listdir(currFolder))
			except:
				dataRemaining = 0

		sid = 0

		remaining = self.num_data * 6
		if (self.num_data == 1000 and not self.train):
			remaining = 101

		for n in range(remaining):
			if dataRemaining == 0:
				i += 1
				try:
					currFolder = self.dataset_folders[i]
				except:
					break
				try:
					dataRemaining = len(os.listdir(currFolder))
				except:
					continue
				sid = 0
			self.data[n] = currFolder + str(sid) + ".pik"

			sid += 1
			dataRemaining -= 1
		self.num_data = len(self.data)
		print(self.num_data)

	def __getitem__(self, idx):
		'''
		returns tuple of 
		'''
		res = None
		with open(self.data[idx], "rb") as f:
			data = pickle.load(f)
			res = (
				data["state_actions"].to(self.device),
				data["id_pair"].to(self.device),
				data["IS_goal_inference"].to(self.device),
				data["IS_action_inference"].to(self.device),
				data["other_agent_true_goal"].to(self.device),
				data['gt_agents'].to(self.device)
			)
			f.close()
		return res

	def __len__(self):
		return self.num_data

def car_state_collate(batch):
	lens = [item[0].shape[0] for item in batch]
	idx = np.argsort(lens)
	idx = list(idx[::-1])

	state_actions = [batch[i][0] for i in idx]
	
	gt_cars = []
	exist_beliefs = []
	telemetry_beliefs = []
	for i in idx:
		exist_tensor = batch[i][1]
		telemetry_tensor = batch[i][2]
		car_exist_tensor = batch[i][3]
		for j in range(len(exist_tensor)):
			exist_beliefs.append(exist_tensor[j])
			telemetry_beliefs.append(telemetry_tensor[j])
			gt_cars.append(car_exist_tensor)
	return state_actions, exist_beliefs, telemetry_beliefs, gt_cars


def car_collate(batch):
	lens = [item[0].shape[0] for item in batch]
	idx = np.argsort(lens)
	idx = list(idx[::-1])

	state_actions = [batch[i][0] for i in idx]
	id_pair = [batch[i][1] for i in idx]

	other_agent_actions = []
	for i, rollout in enumerate(state_actions):
		targetAgent = id_pair[i][1].item()

		for sa_tensor in rollout:
			state, actions = joint_sa_tensor_to_state_action(sa_tensor)
			if targetAgent not in actions:
				other_agent_actions.append(actionIntDict[Action.FORWARD.value])
			else:
				other_agent_actions.append(actionIntDict[actions[targetAgent].value])


	IS_goal_inferences = []
	IS_action_inferences = []
	for i in idx:
		IS_goal_inf = batch[i][2]
		IS_action_inf = batch[i][3]
		for j in range(len(IS_goal_inf)):
			IS_goal_inferences.append(IS_goal_inf[j])
			IS_action_inferences.append(IS_action_inf[j])

	other_agent_goal = torch.cat([batch[i][4] for i in idx], dim=-1)

	return state_actions, id_pair, IS_goal_inferences, IS_action_inferences, other_agent_goal, other_agent_actions








def action_to_string(action):
	return actionStringDict[action.value]


def action_to_one_hot(action):
	action_space = ['forward', 'left', 'right', 'stop', 'signal']
	res = [0] * len(action_space)
	res[action_space.index(action)] = 1
	return res

def one_hot_to_action(one_hot_action):
	action_space = ['forward', 'left', 'right', 'stop', 'signal']
	actionString = action_space[np.argmax(one_hot_action)]
	if actionString == "forward":
		return Action.FORWARD
	elif actionString == "left":
		return Action.LEFT
	elif actionString == "right":
		return Action.RIGHT
	elif actionString == "stop":
		return Action.STOP
	else:
		return Action.SIGNAL


# convert world state, actions to tensor
def state_action_to_joint_tensor(state, actions, device='cpu'):
	'''
	state should be a World object
	actions should be a dictionary of agentID --> action taken as an Action object

	tensor is a [a1_exists, a1_x, a1_y, a1_action_one_hot | ... | an_exists, an_x, an_y, an_action_one_hot | time ]
	'''
	res = []
	num_possible_agents = 16
	num_possible_actions = len(actionStringDict)
	for i in range(num_possible_agents):
		agent = state.agent_exists(i)
		if agent is None:
			exists = 0
			x = 0 
			y = 0 
			heading = 0
			action = action_to_one_hot('forward')  # give an arbitrary action if an agent doesn't exist
		else:
			exists = 1
			x = agent.x 
			y = agent.y
			heading = agent.heading
			action = action_to_one_hot(action_to_string(actions[i]))
		res.append(exists)
		res.append(x)
		res.append(y)
		res.append(heading)
		for a in action:  # add the one hot
			res.append(a)
	res.append(state.t)
	return torch.tensor(res, device=device)

# convert tensor to world state
def joint_sa_tensor_to_state_action(joint_tensor, return_scenario=False):
	# convert tensor to numpy array
	representation = joint_tensor.detach().cpu().numpy()

	# Init the scenario
	scenario = Scenario1()
	state = scenario.w
	state.reset()  # remove all dynamic agents but keep buildings

	state.t = representation[-1]  # we stored time
	
	num_possible_agents = 16
	num_possible_actions = len(actionStringDict)

	single_agent_info_size = 4 + num_possible_actions

	actions = {}

	for i in range(0, len(representation)-1, single_agent_info_size):  # jump in chunks, don't include final time value
		curr = i
		agentID = i // single_agent_info_size
		exists = bool(representation[curr])
		if not exists:
			continue
		curr += 1

		[x, y, heading] = representation[curr: curr+3]
		

		curr += 3

		one_hot = representation[curr:curr+num_possible_actions]
		action = one_hot_to_action(one_hot)
		actions[agentID] = action 

		if i in [0, 1, 2, 3, 8, 9, 10, 11]:
			color = "red"
		else:
			color = "blue"

		agent = Car(Point(x, y), heading, ID=agentID, color=color)
		if action != Action.STOP:
			agent.velocity = Point(220, 0)  # initialize speed,
		state.add(agent)

		actions[i] = action
	if return_scenario:
		return state, actions, scenario
	return state, actions

# convert id pair to tensor
def agent_pair_to_tensor(car1_ID, car2_ID, device='cpu'):
	'''
	car1 and car2 are Car objects
	'''
	return torch.tensor([car1_ID, car2_ID],  device=device)

# convert tensor to id pair
def joint_pair_tensor_to_IDs(tensor_pair):
	return tensor_pair.detach().cpu().numpy()


def get_nn_probs(model, states, idPair):
	"""Get probabilities of desires at every timestep
	from a neural network model ran on a state-action trajectory.

	Args
		model (ToMNet.ToMnet_DesirePred instance)
		states: tensor [num_timesteps, num_rows, num_cols, 2*num_possible_colored_blocks + 4]
			states[t, r, c] is a one hot where
				d=0 -- nothing
				1 -- wall
				2 -- second agent
				3 to (3 + num_colored_blocks - 1) -- colored block type
				3 + num_possible_colored_blocks -- agent
				3 + num_colored_blocks to 3 + 2 * num_colored_blocks -- colored block in inventory
		actions: tensor [num_timesteps, num_actions=5] - one hot representation
			the first action is always [0, 0, 0, 0, 0] -- corresponding to no action
			[1, 0, 0, 0, 0] ... [0, 0, 0, 0, 1] correpond to UP, DOWN, LEFT, RIGHT, PUT_DOWN
			based on envs.construction.Action

	Returns
		probs (np.ndarray [num_timesteps, num_possible_rankings = num_possible_food_trucks!]
					   or [num_timesteps, num_possible_food_trucks])
			where probs[time] is a probability vector
	"""
	# Extract values
	device = actions.device

	# NN predictions
	lens = torch.LongTensor([states.shape[0]]).to('cpu')

	log_prob = model([states], [idPair], lens, last=last)
	probs = torch.softmax(log_prob, 1).cpu().detach().numpy()
	probs /= probs.sum(axis=1, keepdims=True)
	return probs
