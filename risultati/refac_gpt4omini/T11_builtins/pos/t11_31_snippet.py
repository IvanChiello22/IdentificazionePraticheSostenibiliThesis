import numpy as np
from scipy.signal import convolve2d

img = np.random.randn(512, 512)
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
gx = convolve2d(img, sobel_x, mode='same', boundary='symm')
print(gx.shape)