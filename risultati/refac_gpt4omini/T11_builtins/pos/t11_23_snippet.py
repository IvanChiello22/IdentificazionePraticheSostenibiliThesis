import numpy as np
from scipy.signal import convolve2d

def apply_sobel_filter(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return convolve2d(image, sobel_x, mode='same', boundary='symm')

img = np.random.randn(512, 512)
gx = apply_sobel_filter(img)
print(gx.shape)