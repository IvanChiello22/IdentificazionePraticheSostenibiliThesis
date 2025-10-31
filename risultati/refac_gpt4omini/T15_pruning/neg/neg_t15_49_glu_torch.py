import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, 2 * d)

    def forward(self, x):
        a, b = self.W(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)

x = torch.randn(3, 10, 32)
model = GLU(32)

# Use checkpointing to save the model state during training
checkpoint = torch.save(model.state_dict(), 'glu_checkpoint.pth')
output = model(x)
print(output.shape)