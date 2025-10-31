import numpy as np
import tensorflow as tf
import os
import tempfile
import shutil
from tensorflow import keras
from tensorflow.keras import layers

# Snippet 88: mixed set
X = np.random.randn(200, 1000)
k = 7
idx = np.argpartition(-X, k, axis=1)[:, :k]
vals = np.take_along_axis(X, idx, axis=1)
sorted_idx = np.argsort(-vals, axis=1)
topk = np.take_along_axis(vals, sorted_idx, axis=1)
print(topk.shape, round(topk.mean(), 4))

# T19 — Keras: save to tempfile then move (atomic), best‑only by loss
os.makedirs('ckpt', exist_ok=True)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(3)
])
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(128, 20).astype('float32')
y = np.random.randint(0, 3, (128,))
best_loss = float('inf')

for epoch in range(3):
    history = model.fit(x, y, epochs=1, batch_size=64, verbose=0)
    loss = float(history.history['loss'][-1])
    if loss < best_loss:
        best_loss = loss
        fd, temp_file = tempfile.mkstemp(dir='ckpt', prefix='tmp_', suffix='.weights.h5')
        os.close(fd)
        model.save_weights(temp_file)
        shutil.move(temp_file, 'ckpt/best.weights.h5')

print('done')