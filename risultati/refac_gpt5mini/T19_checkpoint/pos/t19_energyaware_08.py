# T19 POS (energy/efficiency-aware checkpointing) — snippet 8
# T19 POS — Keras: rotazione checkpoint con custom Callback (limita file salvati)
import tensorflow as tf, os, glob
from tensorflow import keras; from tensorflow.keras import layers
class RotateCKPT(keras.callbacks.Callback):
    def __init__(self, pattern='ckpt/ep_{epoch:02d}.weights.h5', keep=2): super().__init__(); self.pattern=pattern; self.keep=keep
    def on_epoch_end(self, epoch, logs=None):
        path = self.pattern.format(epoch=epoch+1)
        self.model.save_weights(path)
        hist = sorted(glob.glob('ckpt/ep_*.weights.h5'))
        for f in hist[:-self.keep]: os.remove(f)
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(32, activation='relu', input_shape=(10,)), layers.Dense(2)])
m.compile(optimizer='sgd', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
m.fit(tf.random.normal([128,10]), tf.random.uniform([128], maxval=2, dtype=tf.int32), epochs=4, batch_size=32, callbacks=[RotateCKPT()], verbose=0)
print('done')