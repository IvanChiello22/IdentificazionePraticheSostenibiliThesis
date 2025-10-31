# Snippet 57: triple mix
# T11 — PyTorch: Kronecker product + batched FFT using built‑ins
import torch, torch.fft as fft
A = torch.randn(64, 32); B = torch.randn(32, 16)
K = torch.kron(A, B)
x = torch.randn(8, 1024)
X = fft.rfft(x, dim=-1)
print(K.shape, X.shape)

# T15 — Energy‑aware pruning for MLP neurons (cost ~ fan_in + activation size)
import torch.nn as nn, torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,d=128,h=256): super().__init__(); self.d=d; self.fc1=nn.Linear(d,h); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

def prune_hidden_energy(fc: nn.Linear, keep_ratio, batch_tokens=2048):
    with torch.no_grad():
        imp = fc.weight.detach().abs().mean(dim=1)  # neuron importance
        energy = fc.in_features + batch_tokens  # proxy
        score = imp / (energy + 1e-9)
        k = max(1, int(keep_ratio * fc.out_features))
        topk_idx = torch.topk(score, k).indices
        # create mask of kept neurons and zero out pruned ones (preserve shapes to avoid breaking downstream layers)
        mask = torch.zeros(fc.out_features, dtype=torch.bool, device=fc.weight.device)
        mask[topk_idx] = True
        fc.weight[~mask] = 0
        if fc.bias is not None:
            fc.bias[~mask] = 0
        # store mask as buffer for later inspection; replace if already exists
        try:
            fc.register_buffer('prune_mask', mask)
        except Exception:
            setattr(fc, 'prune_mask', mask)
        # reflect logical out_features as number of kept neurons (keeps original parameter shapes)
        fc.out_features = k

def prune_model_energy(model: nn.Module, keep_ratio, batch_tokens=2048):
    # find Linear layers and compute energy proxy, then prune from most to least energy intensive
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            energy = module.in_features + batch_tokens
            linear_layers.append((energy, name, module))
    linear_layers.sort(key=lambda x: x[0], reverse=True)
    for _, _, layer in linear_layers:
        prune_hidden_energy(layer, keep_ratio, batch_tokens)

m = MLP(); prune_model_energy(m, 0.5, 4096)
x = torch.randn(8, m.d)
# print shape and number of active neurons in fc1 (fall back to out_features if mask absent)
active_fc1 = getattr(m.fc1, 'prune_mask', None)
active_count = int(active_fc1.sum().item()) if active_fc1 is not None else m.fc1.out_features
print(m(x).shape, active_count)

# T19 — Manifest JSON with metrics + size, keep top‑k by metric, delete rest
import os, json, glob
os.makedirs('ckpt', exist_ok=True)
m = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2))
for e in range(1,6):
    loss = float(torch.rand(1))
    path = f'ckpt/e{e:02d}_{loss:.4f}.pt'
    torch.save({'e':e,'loss':loss,'state':m.state_dict()}, path)
    # also keep a latest checkpoint to avoid knowledge loss on premature termination
    torch.save({'e':e,'loss':loss,'state':m.state_dict()}, 'ckpt/latest.pt')
manifest = [{'file':p, 'loss': float(p.split('_')[-1][:-3])} for p in sorted(glob.glob('ckpt/e*.pt'))]
manifest.sort(key=lambda x: x['loss'])
for item in manifest[2:]: os.remove(item['file'])
with open('ckpt/manifest.json','w') as f: json.dump(manifest[:2], f, indent=2)
print('done')