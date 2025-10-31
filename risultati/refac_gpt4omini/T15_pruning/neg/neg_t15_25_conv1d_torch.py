import torch
import torch.nn as nn

# Definizione della rete neurale
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(4, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Creazione dell'istanza della rete e input
net = SimpleConvNet()
x = torch.randn(3, 4, 100)

# Stampa della forma dell'output
print(net(x).shape)