import numpy as np
from scipy.signal import convolve

x = np.random.randn(20000)
k = np.hanning(512)
y = convolve(x, k, mode='full')[:len(x) + len(k) - 1]
print(y.shape)