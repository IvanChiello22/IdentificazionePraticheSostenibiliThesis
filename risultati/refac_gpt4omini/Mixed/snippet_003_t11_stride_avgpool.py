import numpy as np
x = np.random.randn(512, 512)
k = 5
shape = (x.shape[0]-k+1, x.shape[1]-k+1, k, k)
strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1])
windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
avg = windows.mean(axis=(2,3))
print(avg.shape, float(avg.std()))