import torch
import torch.nn as nn
import os
import io
import zipfile

def create_model():
    return nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 2))

def save_checkpoint(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    buf = io.BytesIO()
    torch.save({'model': model.state_dict()}, buf)
    buf.seek(0)
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('state.pt', buf.read())

model = create_model()
save_checkpoint(model, 'ckpt/model_ckpt.zip')
print('done')