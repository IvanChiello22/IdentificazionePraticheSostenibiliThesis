# T19 POS (energy/efficiency-aware checkpointing) — snippet 41
# T19 POS — TensorFlow tf.train.Checkpoint + CheckpointManager (max_to_keep=3)
import tensorflow as tf, os
w = tf.Variable(tf.random.normal([10,2])); b = tf.Variable(tf.zeros([2]))
opt = tf.keras.optimizers.Adam(1e-3); os.makedirs('ckpt_tf', exist_ok=True)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), w=w, b=b, opt=opt)
manager = tf.train.CheckpointManager(ckpt, './ckpt_tf', max_to_keep=3)

# Sustainable tactic: use built-in loss object
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Sustainable tactic: prepare (non-destructive by default) energy-aware pruning utilities
PRUNE_SPARSITY = 0.0  # default 0.0 to preserve behavior; adjust >0.0 to enable pruning

def estimate_energy_intensity(var):
    # simple proxy: number of parameters (larger layers likely more energy intensive)
    return int(tf.size(var).numpy())

def prune_by_magnitude(var, sparsity):
    if sparsity <= 0.0:
        return
    flat = tf.reshape(tf.abs(var), [-1])
    n = int(tf.size(flat).numpy())
    k_keep = max(1, int(round(n * (1.0 - sparsity))))
    values, _ = tf.math.top_k(flat, k=k_keep, sorted=True)
    threshold = values[-1]
    mask = tf.cast(tf.greater_equal(tf.abs(var), threshold), var.dtype)
    var.assign(var * mask)

for e in range(1,7):
    # energy-aware pruning applied starting from most energy intensive layer (no-op if PRUNE_SPARSITY==0)
    layers = [('w', w), ('b', b)]
    layers_sorted = sorted(layers, key=lambda it: estimate_energy_intensity(it[1]), reverse=True)
    top_layer_name, top_layer_var = layers_sorted[0]
    prune_by_magnitude(top_layer_var, PRUNE_SPARSITY)

    x = tf.random.normal([64,10]); y = tf.random.uniform([64], maxval=2, dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = tf.matmul(x,w) + b
        loss = tf.reduce_mean(loss_fn(y, logits))
    grads = tape.gradient(loss, [w,b]); opt.apply_gradients(zip(grads,[w,b]))
    ckpt.step.assign(e); manager.save()
print('done')