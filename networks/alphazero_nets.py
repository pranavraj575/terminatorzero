"""
full networks that go from encoded game (batch size, D1, ..., input_dim) to (policy, value)
"""
import torch
from torch import nn
from networks.transformer import InitialEmbedding, PositionalEncodingLayer, DecoderBlock
from networks.permute import TransToCisPerm, CisToTransPerm
from networks.convNd import ConvBlock, ResBlock
from networks.policy_value_net import PairwisePolicy, CollapsedValue, PolicyValueNet


class AlphaArchitecture(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, moves) -> (torch.Tensor, torch.Tensor):
        """
        note: batch size is kept for legacy, it will probably be 1
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :param moves: iterable (length M) of ((i1, ..., ik), (j1, ..., jk)) or END_TURN queries
        :return: (batch size, M), (batch size, 1): a policy (probability distribution) and a value
        """
        raise NotImplementedError


class AlphaPairwiseCollapseArchitect(AlphaArchitecture):
    """
    uses PairwisePolicy and CollapsedValue to get policy and value from the result sequence
    :param policy_hidden_layers: hidden layers to use to calculate policy
        if None, uses [4*embedding_dim]
    in case of END_TURN being a move:
        :param policy_no_move_collapse_hidden_layers: hidden layers to use to collapse the state
            if None, uses [4*embedding_dim]
        :param policy_no_move_output_hidden_layers: hidden layers to use for outputing probability of END_TURN
            if None, uses [4*embedding_dim]

    :param value_collapse_hidden_layers: hidden layers for collapsing value net
        if None, uses [4*embedding_dim]
    :param value_output_hidden_layers: hidden layers for output in value net
        if None, uses [4*embedding_dim]
    """

    def __init__(self,
                 embedding_dim,
                 policy_hidden_layers=None,
                 policy_no_move_collapse_hidden_layers=None,
                 policy_no_move_output_hidden_layers=None,

                 value_collapse_hidden_layers=None,
                 value_output_hidden_layers=None,
                 ):
        super().__init__()

        if policy_hidden_layers is None:
            policy_hidden_layers = [4*embedding_dim]
        if policy_no_move_collapse_hidden_layers is None:
            policy_no_move_collapse_hidden_layers = [4*embedding_dim]
        if policy_no_move_output_hidden_layers is None:
            policy_no_move_output_hidden_layers = [4*embedding_dim]

        if value_collapse_hidden_layers is None:
            value_collapse_hidden_layers = [4*embedding_dim]
        if value_output_hidden_layers is None:
            value_output_hidden_layers = [4*embedding_dim]

        self.output = PolicyValueNet(
            policy=PairwisePolicy(
                embedding_dim=embedding_dim,
                hidden_layers=policy_hidden_layers,
                no_move_collapse_hidden_layers=policy_no_move_collapse_hidden_layers,
                no_move_output_hidden_layers=policy_no_move_output_hidden_layers,
            ),
            value=CollapsedValue(
                embedding_dim=embedding_dim,
                collapse_hidden_layers=value_collapse_hidden_layers,
                output_hidden_layers=value_output_hidden_layers,
            ),
        )


class ConvolutedArchitect(AlphaPairwiseCollapseArchitect):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 num_residuals,
                 kernel=None,
                 middle_dim=None,

                 policy_hidden_layers=None,
                 policy_no_move_collapse_hidden_layers=None,
                 policy_no_move_output_hidden_layers=None,

                 value_collapse_hidden_layers=None,
                 value_output_hidden_layers=None,
                 ):
        """
        pastes a bunch of CNNs together

        :param input_dim: dimension of input
        :param embedding_dim: dimension to use for embedding
        :param num_residuals: number of residual CNNs to use
        :param kernel: kernel of all convolutions, if None, use (3,3,3,3)
        :param middle_dim: will be sent to all the residual networks as their middle encoding dim

        uses PairwisePolicy and CollapsedValue to get policy and value from the result sequence
        :param policy_hidden_layers: hidden layers to use to calculate policy
            if None, uses [4*embedding_dim]
        in case of END_TURN being a move:
            :param policy_no_move_collapse_hidden_layers: hidden layers to use to collapse the state
                if None, uses [4*embedding_dim]
            :param policy_no_move_output_hidden_layers: hidden layers to use for outputing probability of END_TURN
                if None, uses [4*embedding_dim]

        :param value_collapse_hidden_layers: hidden layers for collapsing value net
            if None, uses [4*embedding_dim]
        :param value_output_hidden_layers: hidden layers for output in value net
            if None, uses [4*embedding_dim]
        """
        super().__init__(
            embedding_dim=embedding_dim,
            policy_hidden_layers=policy_hidden_layers,
            policy_no_move_collapse_hidden_layers=policy_no_move_collapse_hidden_layers,
            policy_no_move_output_hidden_layers=policy_no_move_output_hidden_layers,
            value_collapse_hidden_layers=value_collapse_hidden_layers,
            value_output_hidden_layers=value_output_hidden_layers,
        )
        if kernel is None:
            kernel = (3, 3, 3, 3)
        self.perm1 = TransToCisPerm()
        self.enc = ConvBlock(input_channels=input_dim, output_channels=embedding_dim, kernel=kernel)
        self.layers = nn.ModuleList([
            ResBlock(num_channels=embedding_dim, kernel=kernel, middle_channels=middle_dim) for _ in
            range(num_residuals)
        ])
        # this permutation is nessary for collapsing, as collapse keeps the last dimension
        self.perm2 = CisToTransPerm()

    def forward(self, X: torch.Tensor, moves) -> (torch.Tensor, torch.Tensor):
        """
        note: batch size is kept for legacy, it will probably be 1
        :param X: (batch size, D1, ..., Dk, embedding_dim)
        :param moves: iterable (length M) of ((i1, ..., ik), (j1, ..., jk)) or END_TURN queries
        :return: (batch size, M), (batch size, 1): a policy (probability distribution) and a value
        """
        # X is (batch size, D1, D2, ..., input dim)

        # now (batch size, input dim, D1, D2, ...)
        X = self.perm1(X)

        # (batch size, embedding dim, D1, D2, ...)
        X = self.enc(X)
        for layer in self.layers:
            X = layer(X)

        # (batch size, D1, D2, ..., embedding dim)
        X = self.perm2(X)

        # returns (policy, value)
        return self.output(X, moves)

    def parameters(self, recurse: bool = True):
        for module in (self.enc, self.output):
            for param in module.parameters():
                yield param
        for module in self.layers:
            for param in module.parameters():
                yield param


class TransArchitect(AlphaPairwiseCollapseArchitect):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 num_decoders,
                 n_heads,
                 positional_encoding_nums=None,
                 drop_prob=.2,
                 decoder_hidden_layers=None,

                 policy_hidden_layers=None,
                 policy_no_move_collapse_hidden_layers=None,
                 policy_no_move_output_hidden_layers=None,

                 value_collapse_hidden_layers=None,
                 value_output_hidden_layers=None,
                 ):
        """
        pastes a bunch of decoder blocks together

        data is of shape (batch size, D1, ..., input dim)

        :param input_dim: dim of input
        :param embedding_dim: used by all decoders
        :param num_decoders: number of decoder blocks to use
        :param n_heads: number of attention heads
        :param positional_encoding_nums: number of positional encodings to use on each dimension
            default is (8,8,3,3)
        :param drop_prob: dropout to use in each decoder
        :param decoder_hidden_layers: hidden layers to use in the NN at the end of each decoder
            if None, uses [4*embedding_dim]

        uses PairwisePolicy and CollapsedValue to get policy and value from the result sequence
        :param policy_hidden_layers: hidden layers to use to calculate policy
            if None, uses [4*embedding_dim]
        in case of END_TURN being a move:
            :param policy_no_move_collapse_hidden_layers: hidden layers to use to collapse the state
                if None, uses [4*embedding_dim]
            :param policy_no_move_output_hidden_layers: hidden layers to use for outputing probability of END_TURN
                if None, uses [4*embedding_dim]

        :param value_collapse_hidden_layers: hidden layers for collapsing value net
            if None, uses [4*embedding_dim]
        :param value_output_hidden_layers: hidden layers for output in value net
            if None, uses [4*embedding_dim]
        """
        super().__init__(
            embedding_dim=embedding_dim,
            policy_hidden_layers=policy_hidden_layers,
            policy_no_move_collapse_hidden_layers=policy_no_move_collapse_hidden_layers,
            policy_no_move_output_hidden_layers=policy_no_move_output_hidden_layers,
            value_collapse_hidden_layers=value_collapse_hidden_layers,
            value_output_hidden_layers=value_output_hidden_layers,
        )
        if positional_encoding_nums is None:
            positional_encoding_nums = (8, 8, 3, 3)

        if decoder_hidden_layers is None:
            decoder_hidden_layers = [4*embedding_dim]

        self.emb = InitialEmbedding(
            initial_channels=input_dim,
            embedding_dim=embedding_dim,
            positional_encoding_nums=positional_encoding_nums,
        )
        self.layers = nn.ModuleList([
            DecoderBlock(
                embedding_dim=embedding_dim,
                n_heads=n_heads,
                drop_prob=drop_prob,
                hidden_layers=decoder_hidden_layers,
            )
            for _ in range(num_decoders)
        ])

    def forward(self, X: torch.Tensor, moves) -> (torch.Tensor, torch.Tensor):
        # X is (batch size, D1, ..., initial channels)

        # (batch size, D1, ..., embedding_dim)
        X = self.emb(X)
        for layer in self.layers:
            X = layer(X)

        # returns (policy, value)
        return self.output(X, moves)


class ConvolutedTransArchitect(AlphaPairwiseCollapseArchitect):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 num_blocks,

                 trans_n_heads,

                 cnn_kernel=None,
                 cnn_middle_dim=None,

                 positional_encoding_nums=None,
                 drop_prob=.2,
                 decoder_hidden_layers=None,

                 policy_hidden_layers=None,
                 policy_no_move_collapse_hidden_layers=None,
                 policy_no_move_output_hidden_layers=None,

                 value_collapse_hidden_layers=None,
                 value_output_hidden_layers=None,
                 ):
        """
        pastes a bunch of decoder blocks together

        data is of shape (batch size, D1, ..., input dim)

        :param input_dim: dim of input
        :param embedding_dim: used by all decoders
        :param num_blocks: number of combination CNN decoder blocks to use
        :param trans_n_heads: number of attention heads


        :param cnn_kernel: kernel of all convolutions, if None, use (3,3,3,3)
        :param cnn_middle_dim: will be sent to all the residual networks as their middle encoding dim

        :param positional_encoding_nums: number of positional encodings to use on each dimension
            default is (8,8,3,3)
        :param drop_prob: dropout to use in each decoder
        :param decoder_hidden_layers: hidden layers to use in the NN at the end of each decoder
            if None, uses [4*embedding_dim]

        uses PairwisePolicy and CollapsedValue to get policy and value from the result sequence
        :param policy_hidden_layers: hidden layers to use to calculate policy
            if None, uses [4*embedding_dim]
        in case of END_TURN being a move:
            :param policy_no_move_collapse_hidden_layers: hidden layers to use to collapse the state
                if None, uses [4*embedding_dim]
            :param policy_no_move_output_hidden_layers: hidden layers to use for outputing probability of END_TURN
                if None, uses [4*embedding_dim]

        :param value_collapse_hidden_layers: hidden layers for collapsing value net
            if None, uses [4*embedding_dim]
        :param value_output_hidden_layers: hidden layers for output in value net
            if None, uses [4*embedding_dim]
        """
        super().__init__(
            embedding_dim=embedding_dim,
            policy_hidden_layers=policy_hidden_layers,
            policy_no_move_collapse_hidden_layers=policy_no_move_collapse_hidden_layers,
            policy_no_move_output_hidden_layers=policy_no_move_output_hidden_layers,
            value_collapse_hidden_layers=value_collapse_hidden_layers,
            value_output_hidden_layers=value_output_hidden_layers,
        )
        if positional_encoding_nums is None:
            positional_encoding_nums = (8, 8, 3, 3)
        if decoder_hidden_layers is None:
            decoder_hidden_layers = [4*embedding_dim]

        if cnn_kernel is None:
            cnn_kernel = (3, 3, 3, 3)

        self.emb = InitialEmbedding(
            initial_channels=input_dim,
            embedding_dim=embedding_dim,
            positional_encoding_nums=positional_encoding_nums,
        )
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    n_heads=trans_n_heads,
                    drop_prob=drop_prob,
                    hidden_layers=decoder_hidden_layers,
                )
            )
            self.layers.append(TransToCisPerm())
            self.layers.append(
                ResBlock(
                    num_channels=embedding_dim,
                    kernel=cnn_kernel,
                    middle_channels=cnn_middle_dim,
                )
            )
            self.layers.append(CisToTransPerm())

    def forward(self, X: torch.Tensor, moves) -> (torch.Tensor, torch.Tensor):
        # X is (batch size, D1, ..., initial channels)

        # (batch size, D1, ..., embedding_dim)
        X = self.emb(X)
        for layer in self.layers:
            X = layer(X)

        # returns (policy, value)
        return self.output(X, moves)


if __name__ == '__main__':
    from src.chess5d import Chess5d

    game = Chess5d()

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
    for _ in range(12):
        game.undo_move()
    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)

    alpha = TransArchitect(
        input_dim=encoding.shape[-1],
        embedding_dim=32,
        num_decoders=2,
        n_heads=3,
    )
    alpha = ConvolutedArchitect(
        input_dim=encoding.shape[-1],
        embedding_dim=256,
        num_residuals=32,
    )
    alpha = ConvolutedTransArchitect(
        input_dim=encoding.shape[-1],
        embedding_dim=32,
        num_blocks=2,
        trans_n_heads=3
    )
    print(game)
    print(alpha.forward(encoding, game.all_possible_moves(1)))
