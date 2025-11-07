import numpy as np
from torchlite.nn.module import Module
from torchlite.autograd.ops import mean

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_target):
        assert (
            y_pred.shape == y_target.shape
        ), f"MSELoss: y_pred shape {y_pred.shape} does not match y_target shape {y_target.shape}"

        delta = y_pred - y_target
        squared_delta = delta ** 2

        squared_delta = (y_pred - y_target) ** 2

        loss = mean(squared_delta)
        return loss