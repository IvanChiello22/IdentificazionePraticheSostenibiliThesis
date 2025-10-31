import os
import numpy as np
from scipy.signal import convolve2d
img = np.random.randn(512,512)
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

def compute_sobel_x(image):
    return convolve2d(image, sobel_x, mode='same', boundary='symm')

gx = compute_sobel_x(img)
checkpoint_path = os.getenv('SOBEL_CHECKPOINT')
if checkpoint_path:
    try:
        np.save(checkpoint_path, gx)
    except Exception:
        pass
print(gx.shape)