import itertools
import torch
from torch import nn

from src.chess5d import Chess5d
from networks.ffn import FFN
from networks.permute import CisToTransPerm
from networks.collapse import Collapse


class PositionalEncodingLayer(nn.Module):

    def __init__(self, encoding_nums: [int]):
        super().__init__()
        self.encoding_nums = encoding_nums

    def additional_output(self):
        return 2*sum(self.encoding_nums)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, D1, D2, ..., initial dim)

        The output will have shape (batch_size,  D1, D2, ..., initial_dim + self.additional_output())
        """
        # board_shape=(D1, D2, ...)
        batch_size, *board_shape, initial_dim = X.shape

        for dimension, encoding_num in enumerate(self.encoding_nums):

            append_shape = list(X.shape)
            append_shape[-1] = 2*encoding_num
            # shape is (batch_size, D1, D2, ..., additional size)

            sequence_length = board_shape[dimension]

            # size (sequence length)
            count = torch.arange(0, sequence_length)

            # size (num encodings)
            # frequencies=1/torch.pow(10000,torch.arange(0,self.num_encodings)/self.num_encodings)
            frequencies = 1/torch.pow(2, torch.arange(1, encoding_num + 1))

            # size (sequence length, num encodings)
            inputs = count.view((-1, 1))*(2*torch.pi*frequencies)

            # size (sequence length, 2 * num encodings)
            P = torch.cat((torch.sin(inputs), torch.cos(inputs)), dim=-1)

            # size (1, sequence length, 2 * num encodings)
            P = P.view((1, *P.shape))

            for _ in range(dimension):
                P = P.view(P.shape[0], 1, *P.shape[1:])
            # size (1 (this is the 'batch size' dimension), 1, ..., 1, sequence length, 2 * num encodings)

            # where sequence_length is in the correct dimension that we are looking at

            while len(P.shape) < len(append_shape):
                P = P.view(*P.shape[:-1], 1, P.shape[-1])
            # size (1, 1, ..., 1, sequence length, 1, ..., 2 * num encodings)

            X = torch.cat((X, P + torch.zeros(append_shape)), dim=-1)
            # size (batch_size, D1, D2, ..., X.shape[-1]+2*num encodings)

        return X


class InitialEmbedding(nn.Module):
    """
    initally embeds a chess board (batch size, initial channels, D1, ...)
        into transformer format (batch size, D1, D2, ..., embedding dim)
    """

    def __init__(self, initial_channels, embedding_dim, positional_encoding_nums=None):
        """
        inital channels is the number of channels expected in the embedding
        positional encoding nums are the number of encodings to use for each dimension
            generally, using k encodings will distinguish a 2^k length sequence on that dimension
            default is (8,8,3,3), as 2^3=8 and 2^8 is pretty big
        """
        super().__init__()
        if positional_encoding_nums is None:
            positional_encoding_nums = (8, 8, 3, 3)
        self.perm = CisToTransPerm()
        self.pos_enc = PositionalEncodingLayer(encoding_nums=positional_encoding_nums)
        initial_embedding = initial_channels + self.pos_enc.additional_output()
        self.linear = nn.Linear(initial_embedding, embedding_dim)

    def forward(self, X):
        return self.linear(self.pos_enc(self.perm(X)))


class GeneralAttentionLayer(nn.Module):
    """
    general self attention layer for 5d chess
    """

    def __init__(self, in_dim: int, out_dim: int, cmp_dim=None) -> None:
        super().__init__()
        if cmp_dim is None:
            cmp_dim = out_dim
        self.linear_Q = nn.Linear(in_dim, cmp_dim)
        self.linear_K = nn.Linear(in_dim, cmp_dim)
        self.linear_V = nn.Linear(in_dim, out_dim)

        self.softmax = nn.Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cmp_dim = cmp_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor):
        """
        query_X, key_X and value_X have shape (batch_size, D1, D2, D3, D4, in_dim).

        The dimensions (D1, D2, D3, D4) will probably all be the same

        This will return an output of size (batch_size, D1, D2, D3, D4, out_dim)
        """
        raise NotImplementedError


class SelfAttentionLayerFull(GeneralAttentionLayer):
    """
    self attention layer for 5d chess

    attends all squares to all squares
    """

    def __init__(self, in_dim: int, out_dim: int, cmp_dim=None) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim, cmp_dim=cmp_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor):
        """
        query_X, key_X and value_X have shape (batch_size, D1, D2, D3, D4, in_dim).

        The dimensions (D1, D2, D3, D4) will probably all be the same

        This will return an output of size (batch_size, D1, D2, D3, D4, out_dim)
        """
        # board_size=(D1, D2, D3, D4)
        (batch_size, *board_size, in_dim) = query_X.shape

        # query is (batch_size, D1', D2', D3', D4', comparision dim)
        # key is (batch_size, D1, D2, D3, D4, comparision dim)
        # value is (batch_size, D1, D2, D3, D4, out dim)
        query, key, value = self.linear_Q(query_X), self.linear_K(key_X), self.linear_V(value_X)

        # squish all the middle dimensions
        # (batch_size, D1'*D2'*D3'*D4', comparision dim)
        query = query.view((batch_size, -1, self.cmp_dim))
        # (batch_size, D1*D2*D3*D4, comparision dim)
        key = key.view((batch_size, -1, self.cmp_dim))
        # (batch_size, D1*D2*D3*D4, out dim)
        value = value.view((batch_size, -1, self.out_dim))

        # (batch_size, D1'*D2'*D3'*D4', D1*D2*D3*D4)
        soft_input = torch.bmm(query, key.transpose(1, 2))/(self.cmp_dim**.5)
        att_weights = self.softmax(soft_input)

        return torch.bmm(att_weights, value)


class SelfAttentionLayerSingleMove(GeneralAttentionLayer):
    """
    self attention layer for 5d chess
    since it is far too expensive to compute the attention of each square to every other square
        (O(n^2) in number of squares)
        we only compute the attention between squares that a chess piece can connect in one move
    """

    def __init__(self, in_dim: int, out_dim: int, cmp_dim=None) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim, cmp_dim=cmp_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor):
        """
        query_X, key_X and value_X have shape (batch_size, D1, D2, D3, D4, in_dim).

        The dimensions (D1, D2, D3, D4) will probably all be the same

        This will return an output of size (batch_size, D1, D2, D3, D4, out_dim)
        """
        # board_size=(D1, D2, D3, D4)
        (batch_size, *board_size, in_dim) = query_X.shape

        # query and key are of size (batch_size, D1, D2, D3, D4, comparision dim)
        # value is (batch_size, D1, D2, D3, D4, out dim)
        query, key, value = self.linear_Q(query_X), self.linear_K(key_X), self.linear_V(value_X)

        # shape (batch size, D1, D2, D3, D4, out dim)
        output = torch.zeros((batch_size, *board_size, self.out_dim))

        for idx in itertools.product(*(range(D) for D in board_size)):
            # (batch size, comparision_dim)
            relevant_query = query[:, *idx, :]

            # (batch size, 1, comparision_dim)
            relevant_query = relevant_query.unsqueeze(1)

            # let M be the number of relevant indices
            # Chess5d.connections_of(idx, board_size) is size M x 4
            # relevant_indices is thus 4 x M
            relevant_indices = list(zip(*Chess5d.connections_of(idx, board_size)))
            # relevant_indices=list(zip(*itertools.product(*(range(D) for D in board_size))))

            # (batch size, M, comparision dim)
            relevant_keys = key[:, *relevant_indices, :]
            # (batch size, M, out dim)
            relevant_values = value[:, *relevant_indices, :]

            # (batch size, 1, M)
            soft_input = torch.bmm(relevant_query, relevant_keys.transpose(1, 2))/(self.cmp_dim**.5)
            att_weights = self.softmax(soft_input)

            output[:, *idx, :] = torch.bmm(att_weights, relevant_values)

        return output


class GeneralAttentionToMultiHead(nn.Module):
    def __init__(self, Attention: type[GeneralAttentionLayer], n_heads: int, in_dim: int, out_dim: int, cmp_dim=None):
        super().__init__()

        self.attention_heads = nn.ModuleList([
            Attention(in_dim, out_dim, cmp_dim)
            for _ in range(n_heads)
        ])

        self.linear = nn.Linear(n_heads*out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention

        query_X, key_X and value_X have shape (batch_size, D1, D2, D3, D4, in_dim).

        This will return an output of size (batch_size, D1, D2, D3, D4, out_dim)
        """

        outputs = []
        for attention_head in self.attention_heads:
            out = attention_head(query_X, key_X, value_X)
            # (batch_size, D1, D2, D3, D4, out_dim)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=-1)

        return self.linear(outputs)


class MultiHeadedAttentionSingleMove(GeneralAttentionToMultiHead):
    def __init__(self, n_heads: int, in_dim: int, out_dim: int, cmp_dim=None):
        super().__init__(Attention=SelfAttentionLayerSingleMove,
                         n_heads=n_heads, in_dim=in_dim, out_dim=out_dim, cmp_dim=cmp_dim)


class MultiHeadedAttentionFull(GeneralAttentionToMultiHead):
    def __init__(self, n_heads: int, in_dim: int, out_dim: int, cmp_dim=None):
        super().__init__(Attention=SelfAttentionLayerFull,
                         n_heads=n_heads, in_dim=in_dim, out_dim=out_dim, cmp_dim=cmp_dim)


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, drop_prob=.2, hidden_layers=None):
        super(DecoderBlock, self).__init__()

        self.attention1 = MultiHeadedAttentionSingleMove(n_heads=n_heads,
                                                         in_dim=embedding_dim,
                                                         out_dim=embedding_dim)
        self.attention2 = MultiHeadedAttentionSingleMove(n_heads=n_heads,
                                                         in_dim=embedding_dim,
                                                         out_dim=embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        if hidden_layers is None:
            hidden_layers = [4*embedding_dim]
        self.ffn = FFN(input_dim=embedding_dim, output_dim=embedding_dim, hidden_layers=hidden_layers)

        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, target: torch.Tensor, encoded_source=None) -> torch.Tensor:
        """
        Implementation of a decoder block.

        target has dimensions (batch_size, D1, D2, D3, D4, embedding dim)
        encoded_source (if not None) has dimensions (batch_size, D1', D2', D3', D4', embedding dim)

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, D1, D2, D3, D4, embedding dim) (same as dec)
        """
        _X = target
        X = self.attention1(target, target, target)
        residual = _X + self.dropout1(X)

        X = self.norm1(residual)

        if encoded_source is not None:
            _X = X
            X = self.attention2(X, encoded_source, encoded_source)
            X = self.norm2(_X + self.dropout2(X))

        _X = X
        X = self.ffn(X)

        return self.norm3(_X + self.dropout3(X))


class TransformerArchitect(nn.Module):
    def __init__(self,
                 initial_channels,
                 embedding_dim,
                 num_decoders,
                 n_heads,
                 positional_encoding_nums=None,
                 drop_prob=.2,
                 decoder_hidden_layers=None,
                 collapse_hidden_layers=None,
                 output_hidden_layers=None,
                 ):
        """
        pastes a bunch of transformers together and produces a vector in R^2 (eval for white and eval for black)

        data is of shape (batch size, initial channels, D1, ...)

        num decoders is the number of decoder blocks to use
        decoder_hidden_layers is the hidden layers to use in the NN at the end of each decoder
        drop_prob is the dropout to use in each decoder
        num_heads is the number of attention heads

        embedding dimension is used by all decoders
        positional encoding nums is the number of positional encodings to use on each dimension
            default is (8,8,3,3)

        collapse_hidden_layers and output_hidden_layers are used in collapse and output respectively
            default collapse is just a linear connection
            default output is [4*embedding_layer]
        """
        super().__init__()
        if positional_encoding_nums is None:
            positional_encoding_nums = (8, 8, 3, 3)
        if output_hidden_layers is None:
            output_hidden_layers = [4*embedding_dim]

        self.emb = InitialEmbedding(initial_channels=initial_channels,
                                    embedding_dim=embedding_dim,
                                    positional_encoding_nums=positional_encoding_nums)

        self.blocks = nn.ModuleList([
            DecoderBlock(embedding_dim=embedding_dim,
                         n_heads=n_heads,
                         drop_prob=drop_prob,
                         hidden_layers=decoder_hidden_layers)
            for _ in range(num_decoders)
        ])
        self.collapse = Collapse(embedding_dim=embedding_dim, hidden_layers=collapse_hidden_layers)
        self.output = FFN(input_dim=embedding_dim, output_dim=2, hidden_layers=output_hidden_layers)

    def forward(self, X):
        # (batch size, D1, D2, ..., embedding dim)
        X = self.emb(X)
        for block in self.blocks:
            X = block(X)

        # (batch size, embedding_dim)
        X = self.collapse(X)

        # (batch size, 2)
        return self.output(X)


if __name__ == '__main__':
    import time as timothy

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
    for _ in range(10):
        game.undo_move()
    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)
    encoding = CisToTransPerm()(encoding)

    net = TransformerArchitect(initial_channels=encoding.shape[1],
                               embedding_dim=69,
                               num_decoders=2,
                               n_heads=3,
                               )

    optim = torch.optim.Adam(params=net.parameters())
    start = timothy.time()
    for i in range(10):
        optim.zero_grad()
        out = net.forward(encoding)
        if i == 0:
            print(out.shape)
        loss = torch.nn.MSELoss()(out, torch.ones_like(out))
        loss.backward()
        optim.step()
        if not (i + 1)%1:
            print('time:', int(timothy.time() - start), 'loss:', loss)