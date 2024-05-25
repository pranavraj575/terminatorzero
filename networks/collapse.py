import torch
from torch import nn
from networks.ffn import FFN


class Collapse(nn.Module):
    """
    collapses a sequence down to a single vector

    learns a FFN to determine relevance of each element
    """

    def __init__(self, embedding_dim, hidden_layers=None):
        """
        if hideen layers is none, the FFN learned is a simple linear map
        """
        super().__init__()

        # neural net that ends at a scalar
        self.ffn = FFN(input_dim=embedding_dim, output_dim=1, hidden_layers=hidden_layers)
        self.softmax = nn.Softmax(-1)

    def forward(self, X):
        """
        :param X: (batch size, *, embedding_dim)
        :return: (batch size, embedding_dim)
            for each element of the batch, should be a weighted average of all embeddings
        """
        (batch_size, *middle_dims, embedding_dim) = X.shape
        _X = X
        X = self.ffn(X)
        # X is now (batch size, *, 1)

        # (batch size, M), where M is the product of all the dimensions in *
        X = X.view((batch_size, -1))
        weights = self.softmax(X)

        # (batch size, 1, M)
        weights = weights.unsqueeze(1)

        # (batch size, M, embedding_dim)
        _X = _X.view((batch_size, -1, embedding_dim))

        # (batch size, 1, embedding_dim)
        output = torch.bmm(weights, _X)

        return output.view((batch_size, embedding_dim))
