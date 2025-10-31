import tensorflow as tf
import os

# Initialize variables
weights = tf.Variable(tf.random.normal([10, 2]))
bias = tf.Variable(tf.zeros([2]))
optimizer = tf.keras.optimizers.Adam(1e-3)

# Create checkpoint directory
os.makedirs('ckpt_tf', exist_ok=True)

# Set up checkpoint and manager
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), w=weights, b=bias, opt=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, './ckpt_tf', max_to_keep=3)

# Training loop
for epoch in range(1, 7):
    inputs = tf.random.normal([64, 10])
    labels = tf.random.uniform([64], maxval=2, dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        logits = tf.matmul(inputs, weights) + bias
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    
    gradients = tape.gradient(loss, [weights, bias])
    optimizer.apply_gradients(zip(gradients, [weights, bias]))
    
    checkpoint.step.assign(epoch)
    checkpoint_manager.save()

print('done')