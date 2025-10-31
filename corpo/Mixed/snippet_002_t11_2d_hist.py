# Snippet 2: t11_2d_hist
# T11 — NumPy: 2D histogram + vectorized normalization
import numpy as np
x = np.random.randn(100000); y = 0.5*x + np.random.randn(100000)*0.5
H, xedges, yedges = np.histogram2d(x, y, bins=64, range=[[-4,4],[-4,4]])
P = H / H.sum()
print(P.shape, round(P.max(),4))
