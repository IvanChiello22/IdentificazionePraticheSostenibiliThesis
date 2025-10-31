# Snippet 28: t11_torch_kron_fft + t19_checksum
# T11 — PyTorch: Kronecker product + batched FFT using built‑ins
import torch, torch.fft as fft
A = torch.randn(64, 32); B = torch.randn(32, 16)
K = torch.kron(A, B)
x = torch.randn(8, 1024)
X = fft.rfft(x, dim=-1)
print(K.shape, X.shape)

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
