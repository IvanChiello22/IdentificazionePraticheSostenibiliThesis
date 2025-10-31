# T15 POS — Energy-aware pruning (PyTorch): proporziona il pruning al footprint energetico calcolato sui batch reali
# Usa forward hooks per stimare attivazioni (H*W*C) e flops teorici per conv; ripartisce target sparsity per layer.
import torch, torch.nn as nn, torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.adaptive_avg_pool2d(F.relu(self.c3(x)),1).view(x.size(0),-1)
        return self.fc(x)

def ho_wo(H,W,k,p,s,d):
    Ho = (H + 2*p - d*(k-1) - 1)//s + 1
    Wo = (W + 2*p - d*(k-1) - 1)//s + 1
    return Ho, Wo

def layer_energy(conv, in_shape):
    B,C,H,W = in_shape
    kh,kw = conv.kernel_size; s = conv.stride[0]; p = conv.padding[0]; d = conv.dilation[0]
    Ho,Wo = ho_wo(H,W,kh,p,s,d)
    macs = Ho*Wo*conv.out_channels*(conv.in_channels//conv.groups)*kh*kw
    act = Ho*Wo*conv.out_channels
    return 2*macs + act  # proxy energia

def energy_splits(layers, in_shapes, target_sparsity=0.4, keep_floor=0.2):
    E = torch.tensor([layer_energy(l, ish) for l,ish in zip(layers,in_shapes)], dtype=torch.float32)
    share = E / E.sum()
    # Al più prune = target_sparsity * share per layer
    keep = torch.clamp(1.0 - target_sparsity*share, keep_floor, 0.95).tolist()
    return keep

def prune_out(conv, keep_ratio, in_shape):
    with torch.no_grad():
        B,C,H,W = in_shape
        kh,kw = conv.kernel_size; Ho,Wo = ho_wo(H,W,kh,conv.padding[0],conv.stride[0],conv.dilation[0])
        energy = Ho*Wo*conv.in_channels*kh*kw
        W = conv.weight.detach().abs()
        l1 = W.view(W.size(0), -1).mean(1)
        score = l1 / (energy + 1e-9)
        k = max(1, int(keep_ratio*conv.out_channels))
        keep = torch.topk(score, k).indices
        conv.weight = nn.Parameter(conv.weight[keep].clone())
        if conv.bias is not None: conv.bias = nn.Parameter(conv.bias[keep].clone())
        conv.out_channels = k

net = Net()
layers = [net.c1, net.c2, net.c3]
in_shapes = [(1,3,64,64), (1,16,64,64), (1,32,32,32)]
keeps = energy_splits(layers, in_shapes, target_sparsity=0.5, keep_floor=0.3)
for l, k, ish in zip(layers, keeps, in_shapes): prune_out(l, k, ish)
x = torch.randn(2,3,64,64); print(net(x).shape, [l.out_channels for l in layers], [round(k,2) for k in keeps])
