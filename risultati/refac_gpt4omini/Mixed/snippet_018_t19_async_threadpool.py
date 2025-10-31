import torch, torch.nn as nn, os
from concurrent.futures import ThreadPoolExecutor

os.makedirs('ckpt', exist_ok=True)
model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))

def save_checkpoint(path): 
    torch.save({'state': model.state_dict()}, path)

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(save_checkpoint, f'ckpt/shard_{i}.pt') for i in range(3)]
    _ = [future.result() for future in futures]

print('done')