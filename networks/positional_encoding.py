import torch
from torch import nn


class PositionalEncodingLayer(nn.Module):

    def __init__(self, encoding_nums: [int]):
        super().__init__()
        self.encoding_nums = encoding_nums

    def additional_output(self) -> int:
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


if __name__ == '__main__':
    from src.chess5d import Chess5d
    from src.utilitites import seed_all
    import random

    seed_all()
    game = Chess5d()
    for i in range(20):
        move = random.choice(list(game.all_possible_moves()))
        game.make_move(move)
    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)
    encoding_nums = (3, 2, 5, 4)
    pos_enc = PositionalEncodingLayer(encoding_nums=encoding_nums)
    print(encoding.shape)
    print(pos_enc(encoding).shape)
    for time_slice_sample in pos_enc(encoding)[0, :, 0, 0, 0, :]:
        sample = (time_slice_sample[encoding.shape[-1]:encoding.shape[-1] + 2*encoding_nums[0]])
        sample = torch.round(sample, decimals=2)
        print(sample)
    print()

    for dim_slice_sample in pos_enc(encoding)[0, 0, :, 0, 0, :]:
        sample = (dim_slice_sample[encoding.shape[-1] + 2*encoding_nums[0]:
                                   encoding.shape[-1] + 2*encoding_nums[0] + 2*encoding_nums[1]])
        sample = torch.round(sample, decimals=2)
        print(sample)
