"""
convolutional networks
treats input board (batch size, D1, ..., input_dim) as a 4d array, and convolves appropriately
"""
import torchConvNd
import torch
from torch import nn

from src.chess5d import Chess5d
from networks.permute import TransToCisPerm


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel):
        """
        kernel must be all odd numbers so that dimension stays the same
        """
        super(ConvBlock, self).__init__()
        for k in kernel:
            if not k%2:
                raise Exception('kernel must be only odd numbers')

        stride = [1 for _ in kernel]
        padding = [(k - 1)//2 for k in kernel]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = torchConvNd.ConvNd(input_channels,
                                        output_channels,
                                        list(kernel),
                                        stride=stride,
                                        padding=padding)
        self.conv1_param = nn.ParameterList(self.conv1.parameters())
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu1 = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: shaped (batch size, input channels, D1, D2, ...)
        :return: (batch size, output channels, D1, D2, ...)
        """
        batch_size, input_channels, *other_dimensions = X.shape
        # other_dimensions is (D1, D2, ...)
        X = self.conv1(X)
        X = X.view(batch_size, self.output_channels, -1)
        X = self.bn1(X)
        X = X.view(batch_size, self.output_channels, *other_dimensions)

        return self.relu1(X)


class ResBlock(nn.Module):
    """
    adds residuals to the embedding with CNN
    uses two convolutions and adds the result to the input
    """

    def __init__(self, num_channels: int, kernel, middle_channels=None):
        """
        if middle_channels is None, use num_channels in the middle
        kernel must be all odd numbers so that we can keep the dimensions the same
        """
        super(ResBlock, self).__init__()
        for k in kernel:
            if not k%2:
                raise Exception('kernel must be only odd numbers')

        if middle_channels is None:
            middle_channels = num_channels
        self.num_channels = num_channels
        self.middle_channels = middle_channels
        stride = [1 for _ in kernel]
        padding = [(k - 1)//2 for k in kernel]

        self.conv1 = torchConvNd.ConvNd(num_channels,
                                        middle_channels,
                                        list(kernel),
                                        stride=stride,
                                        padding=padding,
                                        )
        self.conv1_param = nn.ParameterList(self.conv1.parameters())
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = torchConvNd.ConvNd(middle_channels,
                                        num_channels,
                                        list(kernel),
                                        stride=stride,
                                        padding=padding,
                                        )
        self.conv2_param = nn.ParameterList(self.conv2.parameters())
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.relu2 = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: shaped (batch size, input channels, D1, D2, ...)
        :return: (batch size, output channels, D1, D2, ...)
        """
        # (batch size, num channels, D1, D2, ...)
        _X = X

        batch_size, num_channels, *other_dimensions = X.shape
        # other_dimensions is (D1, D2, ...)

        # (batch size, middle channels, D1, D2, ...)
        X = self.conv1(X)
        # (batch size, middle channels, M) where M is the product of the dimensions
        X = X.view(batch_size, self.middle_channels, -1)
        X = self.bn1(X)
        # (batch size, middle channels, D1, D2, ...)
        X = X.view(batch_size, self.middle_channels, *other_dimensions)
        X = self.relu1(X)

        # (batch size, num channels, D1, D2, ...)
        X = self.conv2(X)
        # (batch size, num channels, M)
        X = X.view(batch_size, self.num_channels, -1)
        X = self.bn2(X)
        # (batch size, num channels, D1, D2, ...)
        X = X.view(batch_size, self.num_channels, *other_dimensions)
        return self.relu2(_X + X)


if __name__ == '__main__':

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

    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)

    encoding = TransToCisPerm()(encoding)
    # conv = ConvBlock(encoding.shape[1], 16, (3, 3, 3, 3))
    conv = ResBlock(encoding.shape[1], (3, 3, 3, 3), middle_channels=1)
    optim = torch.optim.Adam(params=conv.parameters())

    for i in range(100):
        optim.zero_grad()
        output = conv(encoding)
        loss = torch.nn.MSELoss()(output, torch.ones_like(output))
        loss.backward()
        optim.step()
        if not (i + 1)%10:
            print(loss)

    print(conv.forward(encoding).shape)
