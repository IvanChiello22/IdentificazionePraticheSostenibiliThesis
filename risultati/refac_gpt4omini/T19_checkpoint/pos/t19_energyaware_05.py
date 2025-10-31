import torch
import torch.nn as nn
import os
import io
import zipfile

def create_checkpoint(model, checkpoint_dir='ckpt', filename='model_ckpt.zip'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    buf = io.BytesIO()
    torch.save({'model': model.state_dict()}, buf)
    buf.seek(0)
    with zipfile.ZipFile(os.path.join(checkpoint_dir, filename), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('state.pt', buf.read())
    print('done')

net = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 2))
create_checkpoint(net)