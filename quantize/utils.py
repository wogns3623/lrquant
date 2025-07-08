import torch
from torch import Tensor


class SELoss(torch.nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor):
        return (input - target) ** 2


def standardize(value: torch.Tensor, dim: torch.Size | int | None = None):
    return (value - value.mean(dim=dim, keepdim=True)) / value.std(
        dim=dim, keepdim=True
    )


class MAFilter(torch.nn.Module):
    def __init__(self, *, replace_method="mean", ma_threshold=100):
        """_summary_

        Args:
            replace_method (str, optional): "zero" or "mean". Defaults to "mean".
            ma_threshold (int, optional): ma_threshold to select massive activation. Defaults to 100.
        """
        super().__init__()
        self.replace_method = replace_method
        self.ma_threshold = ma_threshold

    def forward(self, input: torch.Tensor):
        ma_indices = self.find_massive_activation(input)  # input.shape[:-1]
        if len(ma_indices.nonzero()) == 0:
            return input

        ma = input[ma_indices]  # [ma_count, token_dim]
        ma_standardized = standardize(ma, dim=-1)
        ma_2sd_indices = ma_standardized.abs() >= 2
        ma_2sd_mask = ma_2sd_indices.type(input.dtype)

        # generate & apply mask to ma using ma_indices and ma_2sd_indices
        if self.replace_method == "zero":
            ma_token_scaler = 1 - ma_2sd_mask
        if self.replace_method == "mean":
            # ma_token_scaler = 1 - ma_2sd_mask * (1 - ma_token_scaler)
            # ma_token_scaler = 1 - (1 - ma_token_scaler) * ma_2sd_mask
            # ma_token_scaler = 1 - -(ma_token_scaler - 1) * ma_2sd_mask
            # ma_token_scaler = 1 + (ma_token_scaler - 1) * ma_2sd_mask
            # ma_token_scaler = 1 + (ma_token_scaler * ma_2sd_mask - ma_2sd_mask
            # ma_token_scaler = 1 - ma_2sd_mask + (ma_token_scaler * ma_2sd_mask
            # ma_token_scaler = ma_token_scaler + (ma_token_scaler * ma_2sd_mask

            ma_token_scaler = 1 - ma_2sd_mask

            # # val * ma.mean(dim=1)/val  == ma.mean(dim=1)
            ma_token_scaler_part = ma_2sd_mask / input[ma_indices]
            ma_token_scaler_part *= ma.mean(dim=-1, keepdim=True).expand(ma.shape)
            ma_token_scaler += ma_token_scaler_part

        input_scaler = torch.ones_like(input)
        input_scaler[ma_indices] = ma_token_scaler

        return input * input_scaler

    def find_massive_activation(
        self, value: torch.Tensor, ma_threshold: int | None = None
    ):
        if ma_threshold is None:
            ma_threshold = self.ma_threshold
        magnitudes = value.square().mean(dim=-1)
        ma_filter = magnitudes >= max(ma_threshold, magnitudes.median().item() * 1000)
        return ma_filter
