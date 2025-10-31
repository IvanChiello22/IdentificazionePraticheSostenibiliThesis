import numpy as np
from scipy import ndimage

img = np.random.randn(512, 512)
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
gx = ndimage.convolve(img, sobel_x, mode='mirror')
print(gx.shape)