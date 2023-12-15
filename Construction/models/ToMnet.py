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

	def __init__(self, input_dim, num_channels=64, output_dim=128):
		super(StateEncoder, self).__init__()
		self.input_dim = input_dim
		self.num_channels = num_channels
		self.output_dim = output_dim

		self.conv1 = nn.Conv2d(input_dim[0], num_channels, 1, stride=1)
		self.fc1 = nn.Linear(num_channels * input_dim[1] * input_dim[2], output_dim)

	def forward(self, state):
		state = state.contiguous().view(-1, self.input_dim[0], self.input_dim[1], self.input_dim[2])
		x = F.relu(self.conv1(state.float()))
		x = x.view(-1, self.num_channels * self.input_dim[1] * self.input_dim[2])
		x = F.relu(self.fc1(x))
		return x


class StateActionEncoder(nn.Module):
	"""
	encoder of (state, action) pair
	(state, action) -> state_action_embed
	"""

	def __init__(self, input_dim, action_size, num_channels=64, state_action_embed_dim=128):
		super(StateActionEncoder, self).__init__()

		self.input_dim = input_dim
		self.action_size = action_size
		self.conv_feat_dim = (input_dim[1] - 0) * (input_dim[2] - 0) * num_channels
		self.state_action_embed_dim = state_action_embed_dim
		self.conv1 = nn.Conv2d(input_dim[0] + action_size, num_channels, 1, stride=1)
		self.fc1 = nn.Linear(
			(input_dim[1] - 0) * (input_dim[2] - 0) * num_channels, state_action_embed_dim,
		)
		self.fc2 = nn.Linear(state_action_embed_dim, 2*state_action_embed_dim)
		# self.fc3 = nn.Linear(2*state_action_embed_dim, 2*state_action_embed_dim)
		# self.fc4 = nn.Linear(2*state_action_embed_dim, 2*state_action_embed_dim)
		self.fc5 = nn.Linear(2*state_action_embed_dim, state_action_embed_dim)
	def forward(self, state, action_2d):
		state = state.permute(0, 3, 1, 2).float()
		state_action = torch.cat([state, action_2d.float()], 1)
		x = F.relu(self.conv1(state_action.contiguous()))
		x = x.reshape(-1, self.conv_feat_dim)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# x = F.relu(self.fc3(x))
		# x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		return x


class ToMnet_DesirePred(nn.Module):
	def __init__(self, state_dim, action_size, num_channels, hidden_dim, output_dim, rank=0):
		super(ToMnet_DesirePred, self).__init__()
		self.state_dim = state_dim
		self.action_size = action_size
		self.num_channels = num_channels
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.rank = rank

		self.state_action_encoder = StateActionEncoder(
			state_dim, action_size, num_channels, hidden_dim
		)
		self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
		self.lstm.flatten_parameters()

		self.fc1 = nn.Linear(hidden_dim, 2*hidden_dim)
		# self.fc2 = nn.Linear(2*hidden_dim, 2*hidden_dim)
		self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)

		self.desire_pred = nn.Linear(hidden_dim, output_dim)

	@property
	def device(self):
		return next(self.parameters()).device

	def forward(self, states, actions_2b, lens, last=True):
		batch_size = len(states)
		X = [None] * batch_size
		for sample_id, (s, a) in enumerate(zip(states, actions_2b)):
			X[sample_id] = self.state_action_encoder(s, a)
		# sorted_lens, idx = lens.sort(dim=0, descending=True)

		padded_X = nn.utils.rnn.pad_sequence(X, batch_first=True)
		# sorted_padded_X = padded_X[idx]

		packed_X = nn.utils.rnn.pack_padded_sequence(padded_X, lengths=lens.to("cpu"), batch_first=True)
		# h_0 = Variable(torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(self.device)) #hidden state
		# c_0 = Variable(torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim).to(self.device)) #internal state
		# packed_output, (h, c) = self.lstm(packed_X.float(), (h_0, c_0))
		self.lstm.flatten_parameters()
		packed_output, (h, c) = self.lstm(packed_X.float())

		if last:
			pred = F.relu(self.fc1(h[-1]))
			# pred = F.relu(self.fc2(pred))
			pred = F.relu(self.fc3(pred))
			pred = F.log_softmax(self.desire_pred(pred), dim=-1)
			# pred = F.log_softmax(self.desire_pred(F.relu(self.fc1(h[-1]))), dim=-1)
		else:
			# [batch_size, max_len, hidden_dim]
			padded_output, lens = torch.nn.utils.rnn.pad_packed_sequence(
				packed_output, batch_first=True
			)
			concatenated_output = []
			for b in range(batch_size):
				concatenated_output.append(padded_output[b, : lens[b], :])
			concatenated_output = torch.cat(concatenated_output, dim=0)

			pred = F.relu(self.fc1(concatenated_output))
			# pred = F.relu(self.fc2(pred))
			pred = F.relu(self.fc3(pred))
			pred = F.log_softmax(self.desire_pred(pred), dim=-1)
			# pred = F.log_softmax(self.desire_pred(concatenated_output), dim=-1)

		# max_seq_len = padded_X.size(1)
		# padded_output, _ = nn.utils.rnn.pad_packed_sequence(
		#     packed_output, batch_first=True, total_length=max_seq_len
		# )

		# _, reverse_idx = idx.sort(dim=0, descending=False)
		# padded_output = padded_output[reverse_idx]
		return pred
