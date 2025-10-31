import torch, torch.nn as nn

class PrunedNet(nn.Module):
    def __init__(self):
        super(PrunedNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

net = PrunedNet()
x = torch.randn(2, 256)
print(net(x).shape)