import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """
    state representation, phi(world state):
    """

    def __init__(self, input_dim, num_channels=64, output_dim=6):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(input_dim[0], num_channels, 1, stride=1)
        self.fc1 = nn.Linear(num_channels * input_dim[1] * input_dim[2], output_dim)

    def forward(self, state):
        state = state.contiguous().view(-1, self.input_dim[0], self.input_dim[1], self.input_dim[2])
        x = F.relu(self.conv1(state.float()))
        x = x.view(-1, self.num_channels * self.input_dim[1] * self.input_dim[2])
        x = F.log_softmax(self.fc1(x), dim=-1)
        return x
