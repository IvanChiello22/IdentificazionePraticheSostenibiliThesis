import tensorflow as tf

# Creazione di checkpoint
checkpoint = tf.train.Checkpoint(optimizer=opt, model=w)

w = tf.Variable(tf.random.normal([10, 2]))
b = tf.Variable(tf.zeros([2]))
opt = tf.keras.optimizers.Adam(1e-3)

for e in range(3):
    x = tf.random.normal([32, 10])
    y = tf.random.uniform([32], maxval=2, dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = tf.matmul(x, w) + b
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
    grads = tape.gradient(loss, [w, b])
    opt.apply_gradients(zip(grads, [w, b]))
    
    # Salvataggio del checkpoint
    checkpoint.save(file_prefix='checkpoint')

print('done')