# T19 POS (energy/efficiency-aware checkpointing) — snippet 5
# T19 POS — PyTorch: checkpoint compresso (zip) per ridurre I/O
import torch, torch.nn as nn, os, io, zipfile
net = nn.Sequential(nn.Linear(12,24), nn.ReLU(), nn.Linear(24,2))
os.makedirs('ckpt', exist_ok=True)
buf = io.BytesIO()
torch.save({'model': net.state_dict()}, buf); buf.seek(0)
with zipfile.ZipFile('ckpt/model_ckpt.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.writestr('state.pt', buf.read())
print('done')
