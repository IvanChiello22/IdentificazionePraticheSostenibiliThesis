# Snippet 31: t15_energy_convdepth + t19_atomic
# T15 — Energy‑aware pruning for depthwise‑separable conv (prioritize high‑energy stages)
import torch, torch.nn as nn, torch.nn.functional as F
class DSConv(nn.Module):
    def __init__(self, cin, cout, k=3): super().__init__(); self.dw=nn.Conv2d(cin,cin,k,groups=cin,padding=k//2); self.pw=nn.Conv2d(cin,cout,1)
    def forward(self,x): return self.pw(F.relu(self.dw(x)))
def energy_dw(cin, H, W, k): return 2*H*W*cin*k*k  # depthwise
def energy_pw(cin, cout, H, W): return 2*H*W*cin*cout  # pointwise dominates
def prune_pw_out_energy(pw: nn.Conv2d, keep_ratio, H, W):
    with torch.no_grad():
        e = energy_pw(pw.in_channels, pw.out_channels, H, W)
        imp = pw.weight.detach().abs().view(pw.out_channels, -1).mean(1)
        score = imp/(e+1e-9); k = max(1, int(keep_ratio*pw.out_channels))
        keep = torch.topk(score, k).indices
        pw.weight = nn.Parameter(pw.weight[keep].clone())
        if pw.bias is not None: pw.bias = nn.Parameter(pw.bias[keep].clone())
        pw.out_channels = k

def prune_dsconv_out_energy(dsconv: DSConv, keep_ratio, H, W):
    dw = dsconv.dw
    pw = dsconv.pw
    ksize = dw.kernel_size[0] if isinstance(dw.kernel_size, tuple) else dw.kernel_size
    e_dw = energy_dw(dw.in_channels, H, W, ksize)
    e_pw = energy_pw(pw.in_channels, pw.out_channels, H, W)
    # prune the most energy intensive stage first
    if e_pw >= e_dw:
        prune_pw_out_energy(pw, keep_ratio, H, W)
    else:
        with torch.no_grad():
            imp = dw.weight.detach().abs().view(dw.in_channels, -1).mean(1)
            k = max(1, int(keep_ratio*dw.in_channels))
            keep = torch.topk(imp, k).indices
            keep = keep.sort().values
            # update depthwise conv parameters (weights and optional bias) and channel counts
            dw.weight = nn.Parameter(dw.weight[keep].clone())
            if dw.bias is not None:
                dw.bias = nn.Parameter(dw.bias[keep].clone())
            dw.in_channels = k
            dw.out_channels = k
            dw.groups = k
            # update pointwise conv input channels to match pruned depthwise
            pw.weight = nn.Parameter(pw.weight[:, keep, :, :].clone())
            pw.in_channels = k

net = DSConv(24, 64)
prune_dsconv_out_energy(net, 0.6, 56, 56)
x = torch.randn(1,24,56,56); y = net(x); print(y.shape, net.pw.out_channels)

# T19 — Atomic checkpoint writes (write to tmp then rename), avoid partial files
import torch, torch.nn as nn, os, tempfile, shutil
m = nn.Sequential(nn.Linear(20,40), nn.ReLU(), nn.Linear(40,3)); os.makedirs('ckpt', exist_ok=True)
state = {'model': m.state_dict(), 'meta': {'epoch': 1}}
fd, tmp = tempfile.mkstemp(dir='ckpt', prefix='tmp_', suffix='.pt'); os.close(fd)
try:
    torch.save(state, tmp)
    shutil.move(tmp, 'ckpt/model.pt')
finally:
    if os.path.exists(tmp): os.remove(tmp)
print('done')