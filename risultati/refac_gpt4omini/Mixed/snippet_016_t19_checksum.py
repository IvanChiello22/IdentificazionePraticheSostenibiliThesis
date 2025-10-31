import torch
import torch.nn as nn
import io
import hashlib
import os
import json

def create_model():
    return nn.Sequential(nn.Linear(12, 12), nn.ReLU(), nn.Linear(12, 2))

def save_checkpoint(model, checkpoint_dir='ckpt'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    data = buf.getvalue()
    h = hashlib.sha256(data).hexdigest()
    index = {'sha256': h, 'file': f'{checkpoint_dir}/{h}.pt'}
    
    if not os.path.exists(index['file']):
        with open(index['file'], 'wb') as f:
            f.write(data)
    
    with open(f'{checkpoint_dir}/index.json', 'w') as f:
        json.dump(index, f)
    
    print('done', h[:8])

model = create_model()
save_checkpoint(model)