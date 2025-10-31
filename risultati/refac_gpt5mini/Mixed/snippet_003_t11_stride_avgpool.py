# Snippet 3: t11_stride_avgpool
# T11 — NumPy: sliding-window average via as_strided (no Python loops)
import numpy as np
x = np.random.randn(512, 512)
k = 5
shape = (x.shape[0]-k+1, x.shape[1]-k+1, k, k)
windows = np.lib.stride_tricks.sliding_window_view(x, (k, k))
avg = windows.mean(axis=(2,3))
print(avg.shape, float(avg.std()))