imgs = tf.random.uniform([16, 128, 128, 3])
res = tf.image.resize(imgs, [96, 96], antialias=True)
mean = tf.reduce_mean(res, axis=[1, 2], keepdims=True)
std = tf.math.reduce_std(res, axis=[1, 2], keepdims=True) + 1e-6
norm = (res - mean) / std
print(norm.shape)