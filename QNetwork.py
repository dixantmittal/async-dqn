import torch
import torch.nn as nn

import Logger


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class QNetwork(nn.Module):
    def __init__(self, inDims, outDims):
        super(QNetwork, self).__init__()

        self.inDims = inDims
        self.outDims = outDims

        self.fc = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 7, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=64 * 4, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=outDims)
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 2, 3)
        return self.fc(x)

    def save(self, filename):
        # Switch network to CPU before saving to avoid issues.
        Logger.logger.debug('Saving network to %s', filename)
        torch.save(self.cpu().state_dict(), filename)

    def load(self, filename):
        # Load state dictionary from saved file
        Logger.logger.debug('Loading network from %s', filename)
        self.load_state_dict(torch.load(filename, map_location='cpu'))

    def copy(self, freeze=True):
        # Create a copy of self
        copied = QNetwork(self.inDims, self.outDims)
        copied.load_state_dict(self.state_dict())

        # Freeze its parameters
        if freeze:
            for params in copied.parameters():
                params.requires_grad = False

        return copied
