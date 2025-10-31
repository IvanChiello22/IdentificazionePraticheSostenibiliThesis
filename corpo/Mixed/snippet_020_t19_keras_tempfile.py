# Snippet 20: t19_keras_tempfile
# T19 — Keras: save to tempfile then move (atomic), best‑only by loss
import tensorflow as tf, numpy as np, os, tempfile, shutil
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(128,20).astype('float32'); y = np.random.randint(0,3,(128,))
best = 1e9
for e in range(3):
    hist = m.fit(x,y,epochs=1,batch_size=64,verbose=0)
    loss = float(hist.history['loss'][-1])
    if loss < best:
        best = loss
        fd, tmp = tempfile.mkstemp(dir='ckpt', prefix='tmp_', suffix='.weights.h5'); os.close(fd)
        m.save_weights(tmp); shutil.move(tmp, 'ckpt/best.weights.h5')
print('done')
