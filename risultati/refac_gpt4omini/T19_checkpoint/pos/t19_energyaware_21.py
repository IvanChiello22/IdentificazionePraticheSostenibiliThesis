import torch
import torch.nn as nn
import os
import io
import zipfile

def create_model():
    return nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 2))

def save_checkpoint(model, filepath):
    buf = io.BytesIO()
    torch.save({'model': model.state_dict()}, buf)
    buf.seek(0)
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('state.pt', buf.read())

def main():
    net = create_model()
    os.makedirs('ckpt', exist_ok=True)
    save_checkpoint(net, 'ckpt/model_ckpt.zip')
    print('done')

if __name__ == "__main__":
    main()