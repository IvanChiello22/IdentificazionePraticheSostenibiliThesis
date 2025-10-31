# T11 POS — snippet 42
import numpy as np
x = np.random.randn(20000)
k = np.hanning(512)
n = len(x) + len(k) - 1
y = np.convolve(x, k, mode='full')[:n]
print(y.shape)