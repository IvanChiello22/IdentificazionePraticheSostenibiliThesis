import tensorflow as tf
import os

# Initialize variables
w = tf.Variable(tf.random.normal([10, 2]))
b = tf.Variable(tf.zeros([2]))
opt = tf.keras.optimizers.Adam(1e-3)

# Create checkpoint directory
os.makedirs('ckpt_tf', exist_ok=True)

# Set up checkpoint and manager
ckpt = tf.train.Checkpoint(step=tf.Variable(1), w=w, b=b, opt=opt)
manager = tf.train.CheckpointManager(ckpt, './ckpt_tf', max_to_keep=3)

# Training loop with checkpointing
for e in range(1, 7):
    x = tf.random.normal([64, 10])
    y = tf.random.uniform([64], maxval=2, dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        logits = tf.matmul(x, w) + b
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
    
    grads = tape.gradient(loss, [w, b])
    opt.apply_gradients(zip(grads, [w, b]))
    
    ckpt.step.assign(e)
    manager.save()

print('done')