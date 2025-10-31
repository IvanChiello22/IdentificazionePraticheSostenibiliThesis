# Snippet 21: t11_haversine + t19_atomic
# T11 — NumPy: vectorized Haversine distances (no loops)
import numpy as np
R = 6371.0
lat1 = np.deg2rad(np.random.uniform(-90, 90, (1000,1)))
lon1 = np.deg2rad(np.random.uniform(-180,180, (1000,1)))
lat2 = np.deg2rad(np.random.uniform(-90, 90, (1,1200)))
lon2 = np.deg2rad(np.random.uniform(-180,180, (1,1200)))
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
D = R*c
print(D.shape, D.mean())

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
