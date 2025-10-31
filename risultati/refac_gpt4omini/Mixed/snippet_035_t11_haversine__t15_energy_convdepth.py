import numpy as np
R = 6371.0
lat1 = np.deg2rad(np.random.uniform(-90, 90, (1000,1)))
lon1 = np.deg2rad(np.random.uniform(-180,180, (1000,1)))
lat2 = np.deg2rad(np.random.uniform(-90, 90, (1,1200)))
lon2 = np.deg2rad(np.random.uniform(-180,180, (1,1200)))
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
D = R*c
print(D.shape, D.mean())

import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, groups=cin, padding=k//2)
        self.pw = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.pw(F.relu(self.dw(x)))

def energy_dw(cin, H, W, k):
    return 2 * H * W * cin * k * k  # depthwise

def energy_pw(cin, cout, H, W):
    return 2 * H * W * cin * cout  # pointwise dominates

def prune_pw_out_energy(pw: nn.Conv2d, keep_ratio, H, W):
    with torch.no_grad():
        e = energy_pw(pw.in_channels, pw.out_channels, H, W)
        imp = pw.weight.detach().abs().view(pw.out_channels, -1).mean(1)
        score = imp / (e + 1e-9)
        k = max(1, int(keep_ratio * pw.out_channels))
        keep = torch.topk(score, k).indices
        pw.weight = nn.Parameter(pw.weight[keep].clone())
        if pw.bias is not None:
            pw.bias = nn.Parameter(pw.bias[keep].clone())
        pw.out_channels = k

net = DSConv(24, 64)
prune_pw_out_energy(net.pw, 0.6, 56, 56)
x = torch.randn(1, 24, 56, 56)
y = net(x)
print(y.shape, net.pw.out_channels)