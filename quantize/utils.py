import torch
from torch import Tensor

class SELoss(torch.nn.MSELoss):
  def forward(self, input:Tensor, target: Tensor):
    return (input - target)**2

class TMAFilter(torch.nn.Module):
    def __init__(self, threshold=100):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, input):
        in_token_outlier_indices = self.find_massive_activation(input.squeeze())
        if len(in_token_outlier_indices):
            outlier_tokens = input[0, in_token_outlier_indices][0,0]
            outlier_tokens_standardized = (outlier_tokens-outlier_tokens.mean())/outlier_tokens.std()

            outlier_dimension_2sd_indices = (outlier_tokens_standardized.abs() >= 2).nonzero()
            input[0, in_token_outlier_indices, outlier_dimension_2sd_indices] = outlier_tokens.mean()

        return input
    
    def find_massive_activation(self, values: torch.Tensor):
        token_magnitudes = values.square().mean(dim=1)
        token_outlier_threshold = max(self.threshold, token_magnitudes.median().item()*1000)
        token_outlier_indices = (token_magnitudes >= token_outlier_threshold).nonzero()
        return token_outlier_indices
