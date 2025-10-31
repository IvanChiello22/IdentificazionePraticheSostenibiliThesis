import tensorflow as tf
import torch
import torch.nn as nn
import os
import json
import glob

# Create checkpoint directory
os.makedirs('ckpt', exist_ok=True)

# Define model
model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))

# Training loop with checkpoints
for epoch in range(1, 6):
    loss = float(torch.rand(1))
    checkpoint_path = f'ckpt/e{epoch:02d}_{loss:.4f}.pt'
    torch.save({'e': epoch, 'loss': loss, 'state': model.state_dict()}, checkpoint_path)

# Create manifest of checkpoints
manifest = [{'file': p, 'loss': float(p.split('_')[-1][:-3])} for p in sorted(glob.glob('ckpt/e*.pt'))]
manifest.sort(key=lambda x: x['loss'])

# Keep top-k checkpoints
for item in manifest[2:]:
    os.remove(item['file'])

# Save manifest to JSON
with open('ckpt/manifest.json', 'w') as f:
    json.dump(manifest[:2], f, indent=2)

print('done')

# Image processing
imgs = tf.random.uniform([16, 128, 128, 3])
resized_imgs = tf.image.resize(imgs, [96, 96], antialias=True)
mean = tf.reduce_mean(resized_imgs, axis=[1, 2], keepdims=True)
std = tf.math.reduce_std(resized_imgs, axis=[1, 2], keepdims=True) + 1e-6
normalized_imgs = (resized_imgs - mean) / std
print(normalized_imgs.shape)