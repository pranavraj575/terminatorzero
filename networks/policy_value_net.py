import torch
from torch import nn
from networks.ffn import FFN
from networks.collapse import Collapse
from src.chess5d import END_TURN


class PolicyNet(nn.Module):
    """
    outputs a policy from a (batch size, D1, ..., embedding_dim) encoding
    """

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, moves) -> (torch.Tensor):
        """
        Note: batch size is kept for legacy, it will probably be 1
        Note: must handle the END_TURN move somehow
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :param moves: iterable (length M) of ((i1, ..., ik), (j1, ..., jk)) or END_TURN queries
        :return: (batch size, M) where each M vector is a probability distribution
        """
        raise NotImplementedError


class ValueNet(nn.Module):
    """
    outputs an evaluation from a (batch size, D1, ..., embedding_dim) encoding
    evalutates the game for white's move
    """

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        note: batch size is kept for legacy, it will probably be 1
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :return: (batch size, 1)
        """
        raise NotImplementedError


class PolicyValueNet(nn.Module):
    """
    outputs policy, evaluation from a (batch size, D1, ..., embedding_dim) encoding
    evalutates the game for white's move
    """

    def __init__(self, policy: PolicyNet, value: ValueNet):
        super().__init__()
        self.policy = policy
        self.value = value

    def forward(self, X: torch.Tensor, moves) -> (torch.Tensor, torch.Tensor):
        """
        note: batch size is kept for legacy, it will probably be 1
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :param moves: iterable (length M) of ((i1, ..., ik), (j1, ..., jk)) queries
        :return: (batch size, M), (batch size, 1)
        """
        return self.policy(X, moves), self.value(X)


class PairwisePolicy(PolicyNet):
    """
    outputs a (pick,place) policy from a (batch size, D1, ..., embedding_dim) encoding
    uses a fully connected neural network on the start position and end position to return a probability distribution
    """

    def __init__(self, embedding_dim,
                 hidden_layers=None,
                 no_move_collapse_hidden_layers=None,
                 no_move_output_hidden_layers=None):
        """
        :param embedding_dim: dimenstion of input embedding
        :param hidden_layers: hidden layers to put in between (default None)
        no_move parameters are for if END_TURN is a move
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = FFN(input_dim=2*embedding_dim, output_dim=1, hidden_layers=hidden_layers)
        self.no_move_collapse = Collapse(embedding_dim=embedding_dim, hidden_layers=no_move_collapse_hidden_layers)
        self.no_move_output = FFN(input_dim=embedding_dim, output_dim=1, hidden_layers=no_move_output_hidden_layers)
        self.softmax = nn.Softmax(-1)

    def forward(self, X: torch.Tensor, moves):
        """
        note: batch size is kept for legacy, it will probably be 1
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :param moves: iterable (length M) of ((i1, ..., ik), (j1, ..., jk)) or END_TURN queries
        :return: (batch size, M) where each M vector is a probability distribution
        """
        batch_size, *_, embedding_dim = X.shape
        moves = list(moves)
        M = len(moves)
        pre_softmax = torch.zeros((batch_size, M))

        skip_index = ()
        # set the pre_softmax of END_TURN, if this exists
        if END_TURN in moves:
            important_idx = moves.index(END_TURN)
            skip_index = (important_idx,)
            moves.pop(important_idx)

            # (batch size, embedding dim)
            no_move_vector = self.no_move_collapse(X)
            # (batch size, 1)
            no_move_value = self.no_move_output(no_move_vector)

            pre_softmax[:, important_idx] = no_move_value[:, 0]

        Mp = M - len(skip_index)  # this is M-1 if END_TURN is a move, M otherwise
        # set the pre_softmax of the rest, if there are any
        if Mp > 0:
            start_idxs, end_idxs = zip(*moves)

            input_array = torch.zeros((batch_size, Mp, 2*self.embedding_dim))

            # this will take the start indexes in X and stick them in the top part of input_array
            input_array[:, :, :self.embedding_dim] = X[:, *zip(*start_idxs), :]
            # same for end
            input_array[:, :, self.embedding_dim:] = X[:, *zip(*end_idxs), :]

            # (batch size, Mp, 1)
            output_array = self.network(input_array)

            # if M=Mp, this is just range(M)
            # otherwise, this is range(M) skipping one index (of length Mp)
            normal_indices = (i for i in range(M) if i not in skip_index)

            # these are both of length Mp
            pre_softmax[:, tuple(normal_indices)] = output_array[:, :, 0]

        # (batch size, M)
        return self.softmax(pre_softmax)


class CollapsedValue(ValueNet):
    """
    collapses the sequence into a single vector, then applies a FFN to get scalar output
    """

    def __init__(self,
                 embedding_dim,
                 collapse_hidden_layers=None,
                 output_hidden_layers=None,
                 ):
        super().__init__()
        self.collapse = Collapse(embedding_dim=embedding_dim, hidden_layers=collapse_hidden_layers)
        self.output = FFN(input_dim=embedding_dim, output_dim=1, hidden_layers=output_hidden_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :return: (batch size, 1)
        """

        # (batch size, embedding dim)
        X = self.collapse(X)
        return self.output(X)


if __name__ == '__main__':
    from src.chess5d import Chess5d

    game = Chess5d()

    game.make_move(((0, 0, 1, 3), (0, 0, 3, 3)))
    game.make_move(((1, 0, 6, 4), (1, 0, 4, 4)))
    game.make_move(((2, 0, 0, 1), (0, 0, 2, 1)))
    game.make_move(((1, -1, 6, 6), (1, -1, 5, 6)))
    game.make_move(((2, -1, 1, 7), (2, -1, 2, 7)))
    game.make_move(((3, 0, 6, 6), (3, -1, 6, 6)))
    game.make_move(((4, -1, 2, 1), (4, 0, 4, 1)))
    game.make_move(((5, 0, 4, 4), (5, -1, 4, 4)))
    game.make_move(((6, 0, 4, 1), (6, -1, 4, 3)))
    game.make_move(((7, -1, 7, 1), (7, 0, 5, 1)))
    game.make_move(((8, -1, 0, 1), (8, 0, 2, 1)))
    game.make_move(((9, 0, 5, 1), (9, -1, 7, 1)))
    game.make_move(((10, 0, 2, 1), (10, -1, 0, 1)))
    game.make_move(((11, 0, 7, 3), (11, 0, 3, 7)))
    game.make_move(((11, -1, 7, 1), (11, -1, 5, 2)))
    game.make_move(((12, -1, 4, 3), (8, -1, 4, 2)))
    print(game)
    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)
    print(encoding.shape)
    policy = PairwisePolicy(embedding_dim=encoding.shape[-1], hidden_layers=[3], no_move_collapse_hidden_layers=[8],
                            no_move_output_hidden_layers=[80])
    value = CollapsedValue(embedding_dim=encoding.shape[-1], collapse_hidden_layers=[4], output_hidden_layers=[5])
    pvz = PolicyValueNet(policy=policy, value=value)
    interesting_moves = list(game.all_possible_moves(1))  # contains END_TURN
    optim = torch.optim.Adam(params=pvz.parameters())

    for i in range(1000):
        optim.zero_grad()
        policy, value = pvz(encoding, interesting_moves)
        good_policy = torch.zeros_like(policy)
        good_policy[0, 0] = 1
        loss1 = torch.nn.MSELoss()(policy, good_policy)
        loss2 = torch.nn.MSELoss()(value, torch.ones_like(value))
        loss = loss1 + loss2
        loss.backward()
        optim.step()
        if not i%100:
            print('loss', loss)
            print('policy', policy)
            print('value', value)
