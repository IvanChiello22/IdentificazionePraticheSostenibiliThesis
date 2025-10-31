import torch
import torch.nn as nn
import os
import glob
import json

def create_checkpoint_directory(directory='ckpt'):
    os.makedirs(directory, exist_ok=True)

def save_checkpoint(step, loss, model_state, directory='ckpt'):
    torch.save({'step': step, 'loss': float(loss), 'state': model_state}, f'{directory}/step_{step:03d}.pt')

def prune_checkpoints(directory='ckpt', keep_last_k=3):
    hist = sorted(glob.glob(f'{directory}/step_*.pt'))
    for f in hist[:-keep_last_k]:
        os.remove(f)
    with open(f'{directory}/index.json', 'w') as f:
        json.dump({'last': hist[-keep_last_k:]}, f)

def train_model(steps=10, batch_size=32, input_size=6, hidden_size=12, output_size=2):
    net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
    create_checkpoint_directory()
    
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, input_size)
        y = torch.randint(0, 2, (batch_size,))
        loss = nn.CrossEntropyLoss()(net(x), y)
        save_checkpoint(step, loss, net.state_dict())
        prune_checkpoints()

    print('done')

train_model()