# Snippet 27: t11_tf_image_ops + t19_atomic
# T11 — TensorFlow: vectorized image ops (resize + per‑channel standardize)
import tensorflow as tf
imgs = tf.random.uniform([16, 128, 128, 3])
res = tf.image.resize(imgs, [96, 96], antialias=True)
mean = tf.reduce_mean(res, axis=[1,2], keepdims=True)
std = tf.math.reduce_std(res, axis=[1,2], keepdims=True) + 1e-6
norm = (res - mean) / std
print(norm.shape)

# T19 — Atomic checkpoint writes (write to tmp then rename), avoid partial files
import torch, torch.nn as nn, os, tempfile, shutil
m = nn.Sequential(nn.Linear(20,40), nn.ReLU(), nn.Linear(40,3)); os.makedirs('ckpt', exist_ok=True)
state = {'model': m.state_dict(), 'meta': {'epoch': 1}}
fd, tmp = tempfile.mkstemp(dir='ckpt', prefix='tmp_', suffix='.pt'); os.close(fd)
try:
    torch.save(state, tmp)
    shutil.move(tmp, 'ckpt/model.pt')
finally:
    if os.path.exists(tmp): os.remove(tmp)
print('done')
