import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pdb


class StateEncoder(nn.Module):
	"""
	state representation, phi(world state):
	"""

	def __init__(self, input_dim=145, hidden_dim=128, output_dim=64):
		super(StateEncoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		
		self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
		self.fc3 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
		self.fc4 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
		self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)

	def forward(self, state):
		state = state.float() 
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		return x

class PairEncoder(nn.Module):
	"""
	(agent 1, agent 2) representation of agent 1 inferring about agent 2, phi(world state):
	"""

	def __init__(self, input_dim=2, hidden_dim=32, output_dim=64):
		super(PairEncoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		
		self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

	def forward(self, pair):
		pair = pair.float()
		x = F.relu(self.fc1(pair))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return x

class ToMnet_state_pred(nn.Module):
	'''
	Used to go from partially observed state to belief about state
	'''
	def __init__(self, state_dim=145, hidden_dim=128):
		super(ToMnet_state_pred, self).__init__()
		self.state_dim = state_dim
		self.hidden_dim = hidden_dim
		self.state_encoder = StateEncoder(self.state_dim, hidden_dim=hidden_dim, output_dim=hidden_dim//2)
		self.lstm = nn.LSTM(hidden_dim//2, hidden_dim//2, batch_first=True)
		self.lstm.flatten_parameters()


		# self.existFc1 = nn.Linear(hidden_dim//2, hidden_dim)
		# self.existFc2 = nn.Linear(hidden_dim, hidden_dim)
		# self.existPred = nn.Linear(hidden_dim, 32) 

		# self.telemetryFc1 = nn.Linear(hidden_dim//2, hidden_dim)
		# self.telemetryFc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
		# self.telemetryFc3 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
		# self.telemetryFc4 = nn.Linear(2 * hidden_dim, hidden_dim)
		# self.telementryPred = nn.Linear(hidden_dim, 48)

		self.telemetryFc1 = nn.Linear(hidden_dim//2, hidden_dim)
		self.telemetryFc2 = nn.Linear(hidden_dim, 2*hidden_dim)
		self.telemetryFc3 = nn.Linear(2 * hidden_dim, hidden_dim)
		#self.telemetryFc4 = nn.Linear(2 * hidden_dim, hidden_dim)
		self.telementryPred = nn.Linear(hidden_dim, 48)

	@property
	def device(self):
		return next(self.parameters()).device

	def forward(self, state_action, lens):
		batch_size = len(state_action)
		X = [None] * batch_size
		for sample_id, sa in enumerate(state_action):
			encoded_state = self.state_encoder(sa)
			X[sample_id] = encoded_state
		sorted_lens, idx = lens.sort(dim=0, descending=True)

		padded_X = nn.utils.rnn.pad_sequence(X, batch_first=True)

		packed_X = nn.utils.rnn.pack_padded_sequence(padded_X, lengths=lens.to("cpu"), batch_first=True)

		self.lstm.flatten_parameters()
		packed_output, (h, c) = self.lstm(packed_X.float())

		padded_output, lens = torch.nn.utils.rnn.pad_packed_sequence(
			packed_output, batch_first=True
		)
		concatenated_output = []
		for b in range(batch_size):
			concatenated_output.append(padded_output[b, : lens[b], :])
		concatenated_output = torch.cat(concatenated_output, dim=0)

		# # head 1 - probability the cars exist
		# carExists = F.relu(self.existFc1(concatenated_output))
		# carExists = F.relu(self.existFc2(carExists))
		# carExists = F.log_softmax(self.existPred(carExists), dim=-1)
		# carExists = carExists.view(carExists.shape[0] * 16, 2)
 
		# telemetry prediction
		telemetry = F.relu(self.telemetryFc1(concatenated_output))
		telemetry = F.relu(self.telemetryFc2(telemetry))
		telemetry = F.relu(self.telemetryFc3(telemetry))
		#telemetry = F.relu(self.telemetryFc4(telemetry))
		telemetry = F.relu(self.telementryPred(telemetry))
		telemetry = telemetry.view(telemetry.shape[0] * 16, 3) # (num timesteps * num cars, 3)

		return telemetry

class ToMnet_exist_pred(nn.Module):
	'''
	Used to go from partially observed state to belief about state
	'''
	def __init__(self, state_dim=145, hidden_dim=128):
		super(ToMnet_exist_pred, self).__init__()
		self.state_dim = state_dim
		self.hidden_dim = hidden_dim
		self.state_encoder = StateEncoder(self.state_dim, hidden_dim=hidden_dim, output_dim=hidden_dim//2)
		self.lstm = nn.LSTM(hidden_dim//2, hidden_dim//2, batch_first=True)
		self.lstm.flatten_parameters()


		# self.existFc1 = nn.Linear(hidden_dim//2, hidden_dim)
		# self.existFc2 = nn.Linear(hidden_dim, 2* hidden_dim)
		# self.existFc3 = nn.Linear(2* hidden_dim, 2* hidden_dim)
		# self.existFc4 = nn.Linear(2*hidden_dim, hidden_dim)
		# self.existPred = nn.Linear(hidden_dim, 32) 


		self.existFc1 = nn.Linear(hidden_dim//2, hidden_dim//2)
		self.existFc2 = nn.Linear(hidden_dim//2, hidden_dim)
		#self.existFc3 = nn.Linear(2* hidden_dim, 2* hidden_dim)
		#self.existFc4 = nn.Linear(2*hidden_dim, hidden_dim)
		self.existPred = nn.Linear(hidden_dim, 32) 
		# self.telemetryFc1 = nn.Linear(hidden_dim//2, hidden_dim)
		# self.telemetryFc2 = nn.Linear(hidden_dim, hidden_dim)
		# self.telementryPred = nn.Linear(hidden_dim, 48)

	@property
	def device(self):
		return next(self.parameters()).device

	def forward(self, state_action, lens):
		batch_size = len(state_action)
		X = [None] * batch_size
		for sample_id, sa in enumerate(state_action):
			encoded_state = self.state_encoder(sa)
			X[sample_id] = encoded_state
		sorted_lens, idx = lens.sort(dim=0, descending=True)

		padded_X = nn.utils.rnn.pad_sequence(X, batch_first=True)

		packed_X = nn.utils.rnn.pack_padded_sequence(padded_X, lengths=lens.to("cpu"), batch_first=True)

		self.lstm.flatten_parameters()
		packed_output, (h, c) = self.lstm(packed_X.float())

		padded_output, lens = torch.nn.utils.rnn.pad_packed_sequence(
			packed_output, batch_first=True
		)
		concatenated_output = []
		for b in range(batch_size):
			concatenated_output.append(padded_output[b, : lens[b], :])
		concatenated_output = torch.cat(concatenated_output, dim=0)

		# probability the cars exist
		carExists = F.relu(self.existFc1(concatenated_output))
		carExists = F.relu(self.existFc2(carExists))
		#carExists = F.relu(self.existFc3(carExists))
		#carExists = F.relu(self.existFc4(carExists))
		carExists = self.existPred(carExists)
		carExists = carExists.view(carExists.shape[0] * 16, 2)  # (num timesteps * num cars, 2)
		carExists = F.log_softmax(carExists, dim=-1)
		# # telemetry prediction
		# telemetry = F.relu(self.telemetryFc1(concatenated_output))
		# telemetry = F.relu(self.telemetryFc2(telemetry))
		# telemetry = F.relu(self.telementryPred(telemetry))
		# telemetry = telemetry.view(telemetry.shape[0] * 16, 3)

		return carExists



class ToMnet_car_pred(nn.Module):
	def __init__(self, state_dim=145, hidden_dim=128, output_dim=4):
		super(ToMnet_car_pred, self).__init__()
		self.state_dim = state_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.state_encoder = StateEncoder(self.state_dim, hidden_dim=hidden_dim, output_dim=hidden_dim//2)
		self.pair_encoder = PairEncoder(input_dim=2, hidden_dim=hidden_dim//4, output_dim=hidden_dim//2)

		
		self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
		self.lstm.flatten_parameters()

		self.fc1 = nn.Linear(hidden_dim, 2*hidden_dim)
		self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
		self.action_pred = nn.Linear(hidden_dim, output_dim)

	@property
	def device(self):
		return next(self.parameters()).device

	def forward(self, state_action, inference_pair, lens):
		batch_size = len(state_action)
		X = [None] * batch_size
		for sample_id, sa in enumerate(state_action):
			encoded_state = self.state_encoder(sa)
			pair = inference_pair[sample_id]
			encoded_pair = self.pair_encoder(pair)
			combined = torch.cat((encoded_state, encoded_pair.expand(encoded_state.shape[0], -1)), dim=1)
			X[sample_id] = combined
		sorted_lens, idx = lens.sort(dim=0, descending=True)
		# encoded_state = self.state_encoder(state_action)
		# encoded_pair = self.pair_encoder(inference_pair)
		# X = torch.cat((encoded_state, encoded_pair.expand(encoded_state.shape[0], -1).unsqueeze(1)), dim=1)
		padded_X = nn.utils.rnn.pad_sequence(X, batch_first=True)

		packed_X = nn.utils.rnn.pack_padded_sequence(padded_X, lengths=lens.to("cpu"), batch_first=True)

		self.lstm.flatten_parameters()
		packed_output, (h, c) = self.lstm(packed_X.float())

		padded_output, lens = torch.nn.utils.rnn.pad_packed_sequence(
			packed_output, batch_first=True
		)
		concatenated_output = []
		for b in range(batch_size):
			concatenated_output.append(padded_output[b, : lens[b], :])
		concatenated_output = torch.cat(concatenated_output, dim=0)

		pred = F.relu(self.fc1(concatenated_output))
		pred = F.relu(self.fc2(pred))
		pred = F.log_softmax(self.action_pred(pred), dim=-1)

		return pred