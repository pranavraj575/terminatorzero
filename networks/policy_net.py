import torch
from torch import nn
from networks.ffn import FFN


class PairwisePolicy(nn.Module):
    """
    creates a (pick,place) policy from a (batch size, D1, ..., embedding_dim) encoding
    uses a fully connected neural network on the start position and end position to return a probability distribution
    """

    def __init__(self, embedding_dim, hidden_layers=None):
        """
        :param embedding_dim: dimenstion of input embedding
        :param hidden_layers: hidden layers to put in between (default None)
        """
        self.embedding_dim = embedding_dim
        self.network = FFN(input_dim=2*embedding_dim, output_dim=1, hidden_layers=hidden_layers)
        self.softmax = nn.Softmax(-1)

    def forward(self, X, move_indices):
        """
        note: batch size is kept for legacy, it will probably be 1
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :param move_indices: list (length M) of ((i1, ..., ik), (j1, ..., jk)) queries
        :return: (batch size, M) where each M vector is a probability distribution
        """
        batch_size, *_, embedding_dim = X.shape
        M = len(move_indices)
        start_idxs, end_idxs = zip(*moves)

        input_array = torch.zeros((batch_size, M, 2*self.embedding_dim))

        # this will take the start indexes in X and stick them in the top part of input_array
        input_array[:, :, :self.embedding_dim] = X[:, *zip(start_idxs), :]
        # same for end
        input_array[:, :, self.embedding_dim:] = X[:, *zip(end_idxs), :]

        # (batch size, M, 1)
        pre_softmax = self.network(input_array)
        # (batch size, M)
        pre_softmax = pre_softmax.view((batch_size, M))

        return self.softmax(pre_softmax)
