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
    energy = Ho*Wo*conv.in_channels*kh*kw  # proxy per-output-channel energy
    with torch.no_grad():
        W = conv.weight.detach().abs()
        l1 = W.view(W.size(0), -1).mean(1)
        score = l1 / (energy + 1e-9)
        k = max(1, int(keep_ratio * conv.out_channels))
        keep = torch.topk(score, k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone())
        if conv.bias is not None: conv.bias = nn.Parameter(conv.bias[keep].clone())
        conv.out_channels = k
    return keep

def estimate_energy(conv, in_shape=(1,3,32,32)):
    B,C,H,W = in_shape
    kh,kw = conv.kernel_size
    Ho = (H + 2*conv.padding[0] - conv.dilation[0]*(kh-1) - 1)//conv.stride[0] + 1
    Wo = (W + 2*conv.padding[1] - conv.dilation[1]*(kw-1) - 1)//conv.stride[1] + 1
    return Ho*Wo*conv.in_channels*kh*kw

# Teacher pre-pruning
teacher = TinyNet(); student = TinyNet()

# Checkpoint before pruning (helps recover if interrupted)
torch.save(student.state_dict(), 'student_pre_prune.pt')

# Pruning energy-aware: determine order by estimated energy
layers_info = [
    ('c1', student.c1, (1,3,32,32), 0.75),
    ('c2', student.c2, (1,16,32,32), 0.6),
]
energies = [(name, estimate_energy(conv, in_shape), conv, in_shape, keep) for (name, conv, in_shape, keep) in layers_info]
# sort by descending energy
energies.sort(key=lambda x: x[1], reverse=True)

# Track kept output channel indices for c1 to adjust subsequent conv input channels
kept_indices = {}
for name, _, conv, in_shape, keep_ratio in energies:
    if name == 'c2' and 'c1' in kept_indices:
        # update c2 input channels to match pruned outputs of c1
        keep1 = kept_indices['c1']
        with torch.no_grad():
            # select input channels corresponding to kept outputs of previous layer
            conv.weight = nn.Parameter(conv.weight[:, keep1.clone(), :, :].clone())
            conv.in_channels = len(keep1)
    keep = prune_conv_out_energy(conv, keep_ratio, in_shape)
    kept_indices[name] = keep
    # if we just pruned c1, ensure c2.in_channels matches new c1.out_channels
    if name == 'c1':
        with torch.no_grad():
            student.c2.in_channels = student.c1.out_channels
            # also adapt c2.weight if not already adapted (handles case c2 was pruned earlier)
            if student.c2.weight.size(1) != student.c2.in_channels:
                # If c2.weight has larger in dim, trim; if smaller, pad zeros (unlikely)
                cur_in = student.c2.weight.size(1)
                target_in = student.c2.in_channels
                if cur_in > target_in:
                    student.c2.weight = nn.Parameter(student.c2.weight[:, :target_in, :, :].clone())
                elif cur_in < target_in:
                    pad = torch.zeros(student.c2.weight.size(0), target_in - cur_in, *student.c2.weight.size()[2:], device=student.c2.weight.device)
                    student.c2.weight = nn.Parameter(torch.cat([student.c2.weight, pad], dim=1))

# After pruning, ensure fc matches new c2.out_channels
old_fc = student.fc
new_in = student.c2.out_channels
if old_fc.in_features != new_in:
    new_fc = nn.Linear(new_in, old_fc.out_features)
    with torch.no_grad():
        copy_in = min(old_fc.in_features, new_in)
        new_fc.weight[:, :copy_in] = old_fc.weight[:, :copy_in]
        if old_fc.bias is not None:
            new_fc.bias[:] = old_fc.bias[:]
    student.fc = new_fc

# Checkpoint after pruning / before fine-tune KD
torch.save(student.state_dict(), 'student_pre_kd.pt')

# Light KD fine-tune (demo su batch random)
x = torch.randn(8,3,32,32)
with torch.no_grad(): tlog = teacher(x)
slog = student(x)
loss_kd = F.mse_loss(F.log_softmax(slog,dim=-1), F.log_softmax(tlog,dim=-1))
loss_kd.backward()  # solo dimostrativo

# Checkpoint after KD step
torch.save(student.state_dict(), 'student_post_kd.pt')

print(student(x).shape, 'KD_loss=', float(loss_kd))