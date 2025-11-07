# torchlite/nn/activation.py
import numpy as np
from torchlite.nn.module import Module
from torchlite.autograd.ops import relu

class ReLU(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return relu(x)