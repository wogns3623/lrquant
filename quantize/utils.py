import torch
from torch import Tensor

class SELoss(torch.nn.MSELoss):
  def forward(self, input:Tensor, target: Tensor):
    return (input - target)**2

def detect_outlier_channel(values: torch.Tensor, *, threshold: float=0.1):
    sorted = values.sort(descending=True)
    mean_prev = None
    index = 0
    for i in range(sorted.values.shape[0]):
        sum = sorted.values[:i+1].sum(dim=0)
        mean = sum / (i+1)
        if mean_prev is not None: 
            if (mean_prev-mean).abs() > threshold:
                index = i+1
                break
            
        mean_prev = mean
    return sorted.indices[:index]
