import torch.nn as nn
from block.utils_block import *


channel_list = {1: [128, 256, 512, 23040],
                0.75: [96, 192, 384, 17280],
                0.5: [64, 128, 256, 11520],
                0.25: [32, 64, 128, 5760]}  # [128, 256, 512, 23040]

# [128, 256, 512, 23040]

class ConvNet_2d_uci(nn.Module):
    def __init__(self, channel=[32, 64, 128, 5760]):
        super(ConvNet_2d_uci, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel[0], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(channel[1], channel[2], (6, 1), (3, 1), (1, 0)),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (1, 1), (1, 0))
        )

        self.flatten = FlattenLayer()

        self.classifier = nn.Sequential(
            nn.Linear(channel[3], 6)  # 147456
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    net = ConvNet_2d_uci()

    X = torch.rand(1, 1, 128, 9)


    # X = torch.rand(1, 1, 128, 9)  # pamap2
    # # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)
