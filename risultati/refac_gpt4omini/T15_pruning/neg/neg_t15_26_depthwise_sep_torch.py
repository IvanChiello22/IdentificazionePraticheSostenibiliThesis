import torch, torch.nn as nn

class DepthwiseSeparableConvNet(nn.Module):
    def __init__(self):
        super(DepthwiseSeparableConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(8, 8, 3, groups=8, padding=1),
            nn.Conv2d(8, 16, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

net = DepthwiseSeparableConvNet()
x = torch.randn(1, 8, 32, 32)
print(net(x).shape)