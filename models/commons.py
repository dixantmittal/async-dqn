import torch as t
import torch.nn as nn

hidden_size = 128
d_belief = 128
d_memory = 128


class ResidualLinear(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)

    def forward(self, x):
        y = t.relu(self.linear1(x))
        y = t.relu(self.linear2(y))

        return x + y


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        y = t.relu(self.conv1(x))
        y = t.relu(self.conv2(y))

        return x + y
