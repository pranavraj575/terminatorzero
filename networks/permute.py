import torch
from torch import nn

class CisToTransPerm(nn.Module):
    """
    permutes from convolution order (batch size, channels, D1, D2, ...)
    to transformer order (batch size, D1, D2, ..., channels)
    """

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor):
        # assume X has k+1 dimensions (0, ...,k)
        # this list is (0,2,3,...,k,1)
        return X.permute(0, *range(2, len(X.shape)), 1)


class TransToCisPerm(nn.Module):
    """
    permutes from transformer order (batch size, D1, D2, ..., channels)
    to convolution order (batch size, channels, D1, D2, ...)
    """

    def __init__(self):
        super().__init__()

    def forward(self, X):
        k = len(X.shape) - 1
        # assume X has k+1 dimensions (0, ..., k)
        # this list is (0, k, 1, ..., k-1)

        return X.permute(0, k, range(1, k))