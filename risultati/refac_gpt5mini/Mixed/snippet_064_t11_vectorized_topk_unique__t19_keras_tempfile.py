# Snippet 64: mixed set
# T11 — NumPy: per‑row top‑k without loops via argpartition
import numpy as np
X = np.random.randn(200, 1000)
k = 7
idx = np.argpartition(-X, k, axis=1)[:, :k]
vals = np.take_along_axis(X, idx, axis=1)
sorted_idx = np.argsort(-vals, axis=1)
topk = np.take_along_axis(vals, sorted_idx, axis=1)
print(topk.shape, round(topk.mean(),4))

# T19 — Keras: save to tempfile then move (atomic), best‑only by loss
import tensorflow as tf, numpy as np, os, tempfile, shutil
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(128,20).astype('float32'); y = np.random.randint(0,3,(128,))

class AtomicModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.directory = os.path.dirname(filepath) or '.'
        self.best = float('inf')
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is None:
            return
        if loss < self.best:
            self.best = float(loss)
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix='tmp_', suffix='.weights.h5')
            os.close(fd)
            self.model.save_weights(tmp)
            os.replace(tmp, self.filepath)

ckpt_cb = AtomicModelCheckpoint('ckpt/best.weights.h5')
for e in range(3):
    hist = m.fit(x,y,epochs=1,batch_size=64,verbose=0, callbacks=[ckpt_cb])
print('done')