import torch, torch.nn as nn, torch.nn.functional as F

class MB(nn.Module):
    def __init__(self, in_ch=16, out_ch=32):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return F.relu(self.pw(F.relu(self.dw(x))))

x = torch.randn(1, 16, 64, 64)
model = MB()
checkpoint = model.state_dict()  # Use checkpoint to save model state
print(model(x).shape)