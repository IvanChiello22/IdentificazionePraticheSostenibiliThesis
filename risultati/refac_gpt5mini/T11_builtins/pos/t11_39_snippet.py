import numpy as np
from scipy.ndimage import convolve
img=np.random.randn(512,512);sobel_x=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
gx=convolve(img,sobel_x,mode='reflect')
print(gx.shape)