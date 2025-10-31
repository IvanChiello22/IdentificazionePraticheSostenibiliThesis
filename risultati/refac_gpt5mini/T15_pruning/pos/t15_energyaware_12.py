# T15 POS — Energy-aware pruning con knowledge distillation light per preservare accuracy
# Pruna canali a più alto costo/meno importanti e poi esegue un "fine-tune" distillando dal modello teacher (pre-pruning).
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
        W = conv.weight.detach().abs()
        l1 = W.view(W.size(0), -1).mean(1)
        score = l1 / (energy + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep = torch.topk(score, k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone())
        if conv.bias is not None: conv.bias = nn.Parameter(conv.bias[keep].clone())
        conv.out_channels = k

# Teacher pre-pruning
teacher = TinyNet(); student = TinyNet()
# Pruning energy-aware
prune_conv_out_energy(student.c1, 0.75, (1,3,32,32))
prune_conv_out_energy(student.c2, 0.6, (1,16,32,32))

# Light KD fine-tune (demo su batch random)
x = torch.randn(8,3,32,32)
with torch.no_grad(): tlog = teacher(x)
slog = student(x)
loss_kd = F.mse_loss(F.log_softmax(slog,dim=-1), F.log_softmax(tlog,dim=-1))
loss_kd.backward()  # solo dimostrativo
# Checkpoint per evitare perdita di conoscenza in caso di terminazione prematura
torch.save({'student_state_dict': student.state_dict(), 'kd_loss': float(loss_kd)}, 'student_kd_checkpoint.pth')
print(student(x).shape, 'KD_loss=', float(loss_kd))