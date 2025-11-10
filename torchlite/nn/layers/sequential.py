# torchlite/nn/layers/sequential.py

from torchlite.nn import Module

class Sequential(Module):
  def __init__(self, *args):
    super().__init__()

    for idx, module in enumerate(args):
      if not isinstance(module, Module):
        raise TypeError(f"Argument at {idx} is not an nn.Module.")

      setattr(self, str(idx), module)

  def forward(self, x):  
    for module in self._modules.values():
      x = module(x)
    return x
