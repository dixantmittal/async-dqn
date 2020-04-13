import torch.nn as nn

from models.base import BaseModel
from models.commons import ResidualLinear
from simulator.cart_pole import CartPole


class CartPoleModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=CartPole.state_shape(), out_features=128), nn.ReLU(),
            ResidualLinear(128),
            nn.Linear(in_features=128, out_features=CartPole.n_actions())
        )

    def forward(self, x):
        return self.fc(x)
