import torchConvNd
import torch
from torch import nn
from torch.nn import Parameter

from src.chess5d import Chess5d


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride=None, padding=None):
        super(ConvBlock, self).__init__()
        if stride is None:
            stride = [1 for _ in kernel]
        if padding is None:
            padding = [(k - 1)//2 for k in kernel]
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = torchConvNd.ConvNd(input_channels,
                                        output_channels,
                                        list(kernel),
                                        stride=stride,
                                        padding=padding)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def parameters(self, recurse: bool = True):
        for param in self.bn1.parameters():
            yield param
        for param in self.conv1.parameters():
            yield param

    def forward(self, X):
        """
        :param X: shaped (batch size, input channels, D1, D2, ...)
        :return: (batch size, output channels, D1, D2, ...)
        """
        batch_size, input_channels, *other_dimensions = X.shape
        # other_dimensions is (D1, D2, ...)
        X = self.conv1(X)
        X = self.bn1(X.view(batch_size, self.output_channels, -1))
        return self.relu(X).view(batch_size, self.output_channels, *other_dimensions)


if __name__ == '__main__':

    game = Chess5d()

    game.make_move((0, 0, 1, 3), (0, 0, 3, 3))
    game.make_move((1, 0, 6, 4), (1, 0, 4, 4))
    game.make_move((2, 0, 0, 1), (0, 0, 2, 1))
    game.make_move((1, -1, 6, 6), (1, -1, 5, 6))
    game.make_move((2, -1, 1, 7), (2, -1, 2, 7))
    game.make_move((3, 0, 6, 6), (3, -1, 6, 6))
    game.make_move((4, -1, 2, 1), (4, 0, 4, 1))
    game.make_move((5, 0, 4, 4), (5, -1, 4, 4))
    game.make_move((6, 0, 4, 1), (6, -1, 4, 3))
    game.make_move((7, -1, 7, 1), (7, 0, 5, 1))
    game.make_move((8, -1, 0, 1), (8, 0, 2, 1))
    game.make_move((9, 0, 5, 1), (9, -1, 7, 1))
    game.make_move((10, 0, 2, 1), (10, -1, 0, 1))
    game.make_move((11, 0, 7, 3), (11, 0, 3, 7))
    game.make_move((11, -1, 7, 1), (11, -1, 5, 2))
    game.make_move((12, -1, 4, 3), (8, -1, 4, 2))

    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)

    conv = ConvBlock(19, 16, (3, 3, 3, 3))
    optim = torch.optim.Adam(params=conv.parameters())
    for i in range(1000):
        optim.zero_grad()
        output = conv(encoding)
        loss = torch.nn.MSELoss()(output, torch.ones_like(output))
        loss.backward()
        optim.step()
        if not (i + 1)%10:
            print(loss)

    print(conv.forward(encoding).shape)
