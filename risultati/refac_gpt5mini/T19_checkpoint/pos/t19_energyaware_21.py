import torch
import torch.nn as nn
import os
import io
import zipfile
import tempfile

net = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 2))

def save_compressed_checkpoint(model, ckpt_dir='ckpt', zip_name='model_ckpt.zip', state_name='state.pt'):
    os.makedirs(ckpt_dir, exist_ok=True)
    buf = io.BytesIO()
    try:
        torch.save({'model': model.state_dict()}, buf)
        buf.seek(0)
        fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, suffix='.tmp')
        os.close(fd)
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(state_name, buf.read())
        final_path = os.path.join(ckpt_dir, zip_name)
        os.replace(tmp_path, final_path)
    finally:
        buf.close()

save_compressed_checkpoint(net)
print('done')