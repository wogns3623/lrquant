import torch
from torch import Tensor

class SELoss(torch.nn.MSELoss):
  def forward(self, input:Tensor, target: Tensor):
    return (input - target)**2
