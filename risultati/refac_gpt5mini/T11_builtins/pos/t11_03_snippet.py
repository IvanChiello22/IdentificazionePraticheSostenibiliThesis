import numpy as np
from scipy.signal import convolve2d
rng = np.random.default_rng()
img = rng.standard_normal((512,512))
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
gx = convolve2d(img, sobel_x, mode='same', boundary='symm')
print(gx.shape)