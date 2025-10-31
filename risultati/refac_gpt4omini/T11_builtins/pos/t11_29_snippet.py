import numpy as np
import tensorflow as tf

B, T, D, H = 2, 64, 128, 8
d = D // H
Q = np.random.randn(B, H, T, d)
K = np.random.randn(B, H, T, d)
V = np.random.randn(B, H, T, d)

S = np.einsum('bhtd,bhsd->bhts', Q, K) / np.sqrt(d)
P = np.exp(S - np.max(S, axis=-1, keepdims=True))
P /= np.sum(P, axis=-1, keepdims=True)

O = np.einsum('bhts,bhsd->bhtd', P, V)
print(O.shape)

# Checkpointing
checkpoint_path = "model_checkpoint.ckpt"
checkpoint = tf.train.Checkpoint(Q=Q, K=K, V=V)
checkpoint.save(checkpoint_path)