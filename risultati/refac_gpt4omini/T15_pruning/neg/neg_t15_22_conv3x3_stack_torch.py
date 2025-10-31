import torch, torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

net = SimpleCNN()
x = torch.randn(1, 3, 64, 64)
print(net(x).shape)