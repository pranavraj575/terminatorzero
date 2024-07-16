import torch
from torch import nn
from networks.positional_encoding import PositionalEncodingLayer


class InitialEmbedding(nn.Module):
    """
    initally embeds a chess board (batch size, D1, D2, ..., initial channels)
        into transformer format (batch size, D1, D2, ..., embedding dim)
    """

    def __init__(self, initial_channels: int, embedding_dim: int, positional_encoding_nums=None):
        """
        inital channels is the number of channels expected in the embedding
        positional encoding nums are the number of encodings to use for each dimension
            generally, using k encodings will distinguish a 2^k length sequence on that dimension
            default is (8,8,3,3), as 2^3=8 and 2^8 is pretty big
        """
        super().__init__()
        if positional_encoding_nums is None:
            positional_encoding_nums = (8, 8, 3, 3)
        self.pos_enc = PositionalEncodingLayer(encoding_nums=positional_encoding_nums)
        initial_embedding = initial_channels + self.pos_enc.additional_output()
        self.linear = nn.Linear(initial_embedding, embedding_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(self.pos_enc(X))


class TransformerThing(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 num_decoders,
                 n_heads,
                 positional_encoding_nums=None,
                 drop_prob=.2,
                 ):
        super().__init__()
        self.embed = InitialEmbedding(initial_channels=input_dim,
                                      embedding_dim=embedding_dim,
                                      positional_encoding_nums=positional_encoding_nums)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dropout=drop_prob,
            batch_first=True,
        )
        self.trans = nn.TransformerEncoder(encoder_layer=enc_layer,
                                           num_layers=num_decoders,
                                           )

    def forward(self, X):
        return self.trans(self.embed(X))


encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
print(out.shape)
print(len(list(transformer_encoder.parameters())))
