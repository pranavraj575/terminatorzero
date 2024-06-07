"""
full networks that go from encoded game (batch size, D1, ..., input_dim) to (policy, value)
"""
import torch
from torch import nn

from networks.transformer import InitialEmbedding, DecoderBlock
from networks.permute import TransToCisPerm, CisToTransPerm
from networks.convNd import ConvBlock, ResBlock
from networks.policy_value_net import PairwisePolicy, CollapsedValue, PolicyValueNet
from networks.positional_encoding import PositionalEncodingLayer
from src.chess5d import Chess5d, END_TURN


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
                 positional_encoding_nums=None,
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
        :param positional_encoding_nums: number of positional encodings to use on each dimension
            default is (8,8,3,3)
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

        if positional_encoding_nums is None:
            positional_encoding_nums = (10, 10, 3, 3)
        self.pos_enc = PositionalEncodingLayer(encoding_nums=positional_encoding_nums)
        self.perm1 = TransToCisPerm()

        initial_embedding = input_dim + self.pos_enc.additional_output()
        self.enc = ConvBlock(input_channels=initial_embedding,
                             output_channels=embedding_dim,
                             kernel=kernel,
                             )
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

        # X is (batch size, D1, D2, ..., input embedding)
        X = self.pos_enc(X)

        # now (batch size, input embedding, D1, D2, ...)
        X = self.perm1(X)

        # (batch size, embedding dim, D1, D2, ...)
        X = self.enc(X)
        for layer in self.layers:
            X = layer(X)

        # (batch size, D1, D2, ..., embedding dim)
        X = self.perm2(X)

        # returns (policy, value)
        return self.output(X, moves)


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


def evaluate_network(network: AlphaArchitecture, game, player, moves, chess2d=False):
    if player == 1:
        game.flip_game()
        moves = Chess5d.flip_moves(moves)

    if chess2d:
        # all moves must be (0,0,i,j) in this case
        moves = [END_TURN if move == END_TURN else ((0, 0, *move[0][2:]), (0, 0, *move[1][2:]))
                 for move in moves]
    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)
    policy, value = network.forward(encoding, moves=moves)
    if player == 1:
        game.flip_game()
    return policy, value


if __name__ == '__main__':
    moves = [
        ((0, 0, 1, 3), (0, 0, 3, 3)),
        ((1, 0, 6, 4), (1, 0, 4, 4)),
        ((2, 0, 0, 1), (0, 0, 2, 1)),
        ((1, -1, 6, 6), (1, -1, 5, 6)),
        ((2, -1, 1, 7), (2, -1, 2, 7)),
        ((3, 0, 6, 6), (3, -1, 6, 6)),
        ((4, -1, 2, 1), (4, 0, 4, 1)),
        ((5, 0, 4, 4), (5, -1, 4, 4)),
        ((6, 0, 4, 1), (6, -1, 4, 3)),
        ((7, -1, 7, 1), (7, 0, 5, 1)),
        ((8, -1, 0, 1), (8, 0, 2, 1)),
        ((9, 0, 5, 1), (9, -1, 7, 1)),
        ((10, 0, 2, 1), (10, -1, 0, 1)),
        ((11, 0, 7, 3), (11, 0, 3, 7)),
        ((11, -1, 7, 1), (11, -1, 5, 2)),
        ((12, -1, 4, 3), (8, -1, 4, 2)),
        # ((12, 0, 0, 6), (10, 0, 2, 6)),
        # ((13, 0, 3, 7), (1, 0, 3, 7)),
        # ((2, 1, 0, 1), (6, 0, 0, 1)),
    ]

    game = Chess5d()

    for move in moves:
        game.make_move(move)
    for _ in range(12):
        game.undo_move()
    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)

    alpha = TransArchitect(
        input_dim=encoding.shape[-1],
        embedding_dim=256,
        num_decoders=5,
        n_heads=3,
    )
    """
    alpha = ConvolutedArchitect(
        input_dim=encoding.shape[-1],
        embedding_dim=256,
        num_residuals=5,
    )
    alpha = ConvolutedTransArchitect(
        input_dim=encoding.shape[-1],
        embedding_dim=32,
        num_blocks=2,
        trans_n_heads=3
    )
    """
    print(game)
    print(alpha.forward(encoding, game.all_possible_moves(1)))

    from src.chess5d import BOARD_SIZE, Board, EMPTY, as_player, KING, QUEEN

    left = (BOARD_SIZE - 2)//2
    board = Board(pieces=[[EMPTY for _ in range(left)] +
                          [as_player(KING, 0), as_player(QUEEN, 0)] +
                          [EMPTY for _ in range(BOARD_SIZE - 2 - left)]] +
                         [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 2)] +
                         [[EMPTY for _ in range(left)] +
                          [as_player(KING, 1)] +
                          [EMPTY for _ in range(BOARD_SIZE - 1 - left)]]
                  )
    test_games = []
    test_games.append((Chess5d(initial_board=board.clone()), 9))
    board = board.flipped_board()
    test_games.append((Chess5d(initial_board=board.clone()), -9))
    board.set_player(1 - board.player)
    test_games.append((Chess5d(initial_board=board.clone()), -8))
    board = board.flipped_board()
    test_games.append((Chess5d(initial_board=board.clone()), 8))

    inputs = [torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0) for game, value in test_games]
    optim = torch.optim.Adam(alpha.parameters(), )
    losses = []
    for epoch in range(1000):
        optim.zero_grad()
        disp = not epoch%5
        loss = torch.zeros(1)
        for enc, (game, value) in zip(inputs, test_games):
            _, guess_val = alpha.forward(enc, moves=[])
            if disp:
                print('value', value, 'guess', guess_val)
            loss += torch.square(value - guess_val.flatten())
        loss = loss
        loss.backward()
        losses.append(loss.item())
        optim.step()
        if disp:
            print('loss', loss)
            print('epoch', epoch, 'done')
            print()
    from matplotlib import pyplot as plt

    plt.plot(losses)
    plt.show()
