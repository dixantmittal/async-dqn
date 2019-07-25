import torch
import torch.nn as nn

import Logger


class QNetwork(nn.Module):
    def __init__(self, inDims, outDims):
        super(QNetwork, self).__init__()

        self.inDims = inDims
        self.outDims = outDims

        self.fc = nn.Sequential(
            nn.Linear(in_features=inDims, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=outDims)
        )

    def forward(self, x):
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
