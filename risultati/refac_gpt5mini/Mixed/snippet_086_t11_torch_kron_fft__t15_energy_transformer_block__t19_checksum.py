# Snippet 86: mixed set
# T11 — PyTorch: Kronecker product + batched FFT using built‑ins
import torch, torch.fft as fft
A = torch.randn(64, 32); B = torch.randn(32, 16)
K = torch.kron(A, B)
x = torch.randn(8, 1024)
X = fft.rfft(x, dim=-1)
print(K.shape, X.shape)

# T15 — Energy‑aware pruning for Transformer block (heads + FFN width)
import torch, torch.nn as nn, torch.nn.functional as F
class Block(nn.Module):
    def __init__(self, d=96, h=6, ffn=192):
        super().__init__(); self.h=h; self.dk=d//h
        self.qkv=nn.Linear(d, 3*d); self.o=nn.Linear(d,d)
        self.fc1=nn.Linear(d, ffn); self.fc2=nn.Linear(ffn, d)
    def forward(self,x):
        B,T,D = x.shape
        qkv=self.qkv(x); q,k,v = qkv.chunk(3, dim=-1)
        q=q.view(B,T,self.h,self.dk).transpose(1,2); k=k.view(B,T,self.h,self.dk).transpose(1,2); v=v.view(B,T,self.h,self.dk).transpose(1,2)
        a=(q@k.transpose(-2,-1))/self.dk**0.5; p=a.softmax(-1); o=(p@v).transpose(1,2).reshape(B,T,-1)
        x = self.o(o) + x
        return self.fc2(F.gelu(self.fc1(x)))
def head_energy(T, dk): return T*T*dk + T*T + T*dk
def prune_heads_energy(o: nn.Linear, h, dk, keep_ratio, T):
    with torch.no_grad():
        H=h; D=H*dk; W=o.weight.detach()
        blocks = W[:, :D].view(W.size(0), H, dk)
        imp = blocks.abs().mean(dim=(0,2))  # per-head importance
        e = torch.full_like(imp, float(head_energy(T, dk)))
        score = imp/(e+1e-9); k=max(1,int(keep_ratio*H))
        keep = torch.topk(score,k).indices.tolist()
        new_in = dk*k
        o.weight = nn.Parameter(W[:, :new_in].clone())
        if o.bias is not None: o.bias = nn.Parameter(o.bias.clone())
        return k
def prune_ffn_energy(fc1: nn.Linear, keep_ratio, tokens):
    with torch.no_grad():
        imp = fc1.weight.detach().abs().mean(1)
        e = fc1.in_features + tokens
        k = max(1,int(keep_ratio*fc1.out_features))
        keep = torch.topk(imp/(e+1e-9), k).indices
        fc1.weight = nn.Parameter(fc1.weight[keep].clone())
        if fc1.bias is not None: fc1.bias = nn.Parameter(fc1.bias[keep].clone())
        fc1.out_features = k

def prune_model_energy_order(block: Block, keep_heads_ratio: float, keep_ffn_ratio: float, T: int, tokens: int):
    # estimate total energy for heads vs ffn and prune the more costly first
    head_total = head_energy(T, block.dk) * block.h
    ffn_total = block.fc1.in_features * block.fc1.out_features + tokens
    if head_total >= ffn_total:
        new_h = prune_heads_energy(block.o, block.h, block.dk, keep_heads_ratio, T)
        prune_ffn_energy(block.fc1, keep_ffn_ratio, tokens)
    else:
        prune_ffn_energy(block.fc1, keep_ffn_ratio, tokens)
        new_h = prune_heads_energy(block.o, block.h, block.dk, keep_heads_ratio, T)
    return new_h

blk = Block()
new_h = prune_model_energy_order(blk, 0.5, 0.6, T=64, tokens=4096)

# checkpoint blk state_dict (deduplicate by sha256) to avoid knowledge loss
import io, hashlib, os, json
os.makedirs('ckpt', exist_ok=True)
buf = io.BytesIO(); torch.save(blk.state_dict(), buf); data = buf.getvalue()
h_blk = hashlib.sha256(data).hexdigest()
index_blk = {'sha256': h_blk, 'file': f'ckpt/{h_blk}.pt', 'model': 'blk'}
if not os.path.exists(index_blk['file']):
    with open(index_blk['file'], 'wb') as f: f.write(data)
with open('ckpt/index_blk.json','w') as f: json.dump(index_blk, f)

x = torch.randn(2,64,96); print(blk(x).shape, 'heads', new_h, 'ffn', blk.fc1.out_features)

# T19 — Deduplicate checkpoints by SHA256 (skip identical content)
import torch, torch.nn as nn, io, hashlib, os, json
m = nn.Sequential(nn.Linear(12,12), nn.ReLU(), nn.Linear(12,2)); os.makedirs('ckpt', exist_ok=True)
buf = io.BytesIO(); torch.save(m.state_dict(), buf); data = buf.getvalue()
h = hashlib.sha256(data).hexdigest()
index = {'sha256': h, 'file': f'ckpt/{h}.pt'}
if not os.path.exists(index['file']):
    with open(index['file'], 'wb') as f: f.write(data)
with open('ckpt/index.json','w') as f: json.dump(index, f)
print('done', h[:8])