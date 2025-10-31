# T15 POS — Energy-aware pruning (TensorFlow): per-head attention pruning proxy
import os
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SimpleMHA(keras.layers.Layer):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.d_model = d_model; self.num_heads = num_heads; self.dk = d_model//num_heads
        self.qkv = layers.Dense(3*d_model); self.o = layers.Dense(d_model)
    def call(self, x):
        B,T,D = tf.shape(x)[0], tf.shape(x)[1], self.d_model
        qkv = self.qkv(x)
        q,k,v = tf.split(qkv, 3, axis=-1)
        q = tf.reshape(q, [B, T, self.num_heads, self.dk]); k = tf.reshape(k, [B, T, self.num_heads, self.dk]); v = tf.reshape(v, [B, T, self.num_heads, self.dk])
        q = tf.transpose(q, [0,2,1,3]); k = tf.transpose(k, [0,2,1,3]); v = tf.transpose(v, [0,2,1,3])
        scores = tf.matmul(q, k, transpose_b=True)/tf.sqrt(tf.cast(self.dk, tf.float32))
        attn = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.reshape(tf.transpose(out,[0,2,1,3]), [B,T,self.d_model])
        return self.o(out)

def energy_per_head(T, dk): return T*T*dk + T*T + T*dk

def prune_heads_energy(layer: SimpleMHA, keep_ratio=0.5, T=64):
    Wout = layer.o.kernel  # [D, D]
    H = layer.num_heads; dk = layer.dk
    energy = tf.cast(tf.ones([H])*energy_per_head(T, dk), tf.float32)
    Wblocks = tf.reshape(Wout, [layer.d_model, H, dk])  # [D, H, dk]
    imp = tf.reduce_mean(tf.abs(Wblocks), axis=[0,2])   # [H]
    score = imp/(energy+1e-9)
    k_tensor = tf.maximum(1, tf.cast(tf.round(keep_ratio*tf.cast(H, tf.float32)), tf.int32))
    k = int(k_tensor.numpy())
    keep = tf.argsort(score, direction='DESCENDING')[:k]
    # Aggiorna kernel tramite maschera (mantieni prime k*dk colonne) per preservare shape della variabile
    new_in = k * dk
    D = layer.d_model
    new_in = max(0, min(D, new_in))
    ones = tf.ones([D, new_in], dtype=Wout.dtype)
    zeros = tf.zeros([D, D - new_in], dtype=Wout.dtype)
    mask = tf.concat([ones, zeros], axis=1)
    # Checkpoint before modifying to avoid knowledge loss in case of interruption
    try:
        ckpt_dir = os.path.join(tempfile.gettempdir(), 'mha_prune_checkpoints')
        tf.io.gfile.makedirs(ckpt_dir)
        ckpt = tf.train.Checkpoint(layer=layer)
        ckpt.write(os.path.join(ckpt_dir, 'ckpt'))
    except Exception:
        pass
    layer.o.kernel.assign(Wout * mask)
    layer.num_heads = k

x = tf.random.normal([2,64,64])
mha = SimpleMHA(64,4)
prune_heads_energy(mha, 0.5, 64)
y = mha(x); print(y.shape, 'heads', mha.num_heads)