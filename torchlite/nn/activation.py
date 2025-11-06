# torchlite/nn/activation.py
import numpy as np
from torchlite.nn.module import Module

class ReLU(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return np.max(0, x)