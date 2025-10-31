import torch, torch.nn as nn, torch.nn.functional as F

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)
    def forward(self,x):
        x = F.relu(self.c1(x)); x = F.adaptive_avg_pool2d(F.relu(self.c2(x)),1).view(x.size(0),-1); return self.fc(x)

def prune_conv_out_energy(conv, keep_ratio, in_shape=(1,3,32,32)):
    B,C,H,W = in_shape
    kh,kw = conv.kernel_size
    Ho = (H + 2*conv.padding[0] - conv.dilation[0]*(kh-1) - 1)//conv.stride[0] + 1
    Wo = (W + 2*conv.padding[1] - conv.dilation[1]*(kw-1) - 1)//conv.stride[1] + 1
    energy = Ho*Wo*conv.in_channels*kh*kw  # proxy por-canale output (costante per canale)
    with torch.no_grad():
        Wt = conv.weight.detach().abs()
        l1 = Wt.view(Wt.size(0), -1).mean(1)
        score = l1 / (energy + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep = torch.topk(score, k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone())
        if conv.bias is not None:
            conv.bias = nn.Parameter(conv.bias[keep].clone())
        conv.out_channels = k
    return keep

def _compute_conv_energy(conv, in_shape=(1,3,32,32)):
    B,C,H,W = in_shape
    kh,kw = conv.kernel_size
    Ho = (H + 2*conv.padding[0] - conv.dilation[0]*(kh-1) - 1)//conv.stride[0] + 1
    Wo = (W + 2*conv.padding[1] - conv.dilation[1]*(kw-1) - 1)//conv.stride[1] + 1
    return Ho*Wo*conv.in_channels*kh*kw

# Teacher pre-pruning
teacher = TinyNet(); student = TinyNet()

# Save checkpoint before pruning
torch.save({'teacher_state': teacher.state_dict(), 'student_state': student.state_dict()}, 'checkpoint_before_prune.pth')

# Determine energy per layer and prune from most to least energy intensive
energies = {
    'c1': _compute_conv_energy(student.c1, (1,3,32,32)),
    'c2': _compute_conv_energy(student.c2, (1,16,32,32))
}
order = sorted(energies.items(), key=lambda x: x[1], reverse=True)
keeps = {}

for name, _ in order:
    if name == 'c1':
        keeps['c1'] = prune_conv_out_energy(student.c1, 0.75, (1,3,32,32))
    elif name == 'c2':
        keeps['c2'] = prune_conv_out_energy(student.c2, 0.6, (1,16,32,32))

# Align subsequent layer input channels and final FC with pruned outputs
if 'c1' in keeps:
    keep1 = keeps['c1']
    # Adjust c2 input channels to match pruned c1 outputs
    with torch.no_grad():
        student.c2.weight = nn.Parameter(student.c2.weight[:, keep1, :, :].clone())
        student.c2.in_channels = student.c2.weight.size(1)

if 'c2' in keeps:
    keep2 = keeps['c2']
    # Adjust fc input features to match pruned c2 outputs
    with torch.no_grad():
        student.fc.weight = nn.Parameter(student.fc.weight[:, keep2].clone())
        student.fc.in_features = student.fc.weight.size(1)

# Save checkpoint after pruning
torch.save({'teacher_state': teacher.state_dict(), 'student_state': student.state_dict()}, 'checkpoint_after_prune.pth')

# Light KD fine-tune (demo su batch random)
x = torch.randn(8,3,32,32)
with torch.no_grad(): tlog = teacher(x)
slog = student(x)
loss_kd = F.mse_loss(F.log_softmax(slog,dim=-1), F.log_softmax(tlog,dim=-1))
loss_kd.backward()  # solo dimostrativo

# Save checkpoint after KD step
torch.save({'teacher_state': teacher.state_dict(), 'student_state': student.state_dict(), 'loss': loss_kd.item()}, 'checkpoint_after_kd.pth')

print(student(x).shape, 'KD_loss=', float(loss_kd))