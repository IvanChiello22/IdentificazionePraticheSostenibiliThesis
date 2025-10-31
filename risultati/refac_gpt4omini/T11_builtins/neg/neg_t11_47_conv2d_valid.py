image = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]
kernel = [
    [1, 0],
    [0, -1]
]

import numpy as np

image_array = np.array(image)
kernel_array = np.array(kernel)
out = np.convolve(image_array.flatten(), kernel_array.flatten(), mode='valid').reshape(image_array.shape[0] - kernel_array.shape[0] + 1, image_array.shape[1] - kernel_array.shape[1] + 1).tolist()

print(out)