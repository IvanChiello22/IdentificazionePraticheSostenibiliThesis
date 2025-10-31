inp = [1.0, 2.0, -1.0]
W = [[0.1, -0.3],
     [0.5,  0.2],
     [0.0,  0.4]]
b = [0.01, -0.02]

import numpy as np

out = np.dot(inp, np.array(W).T) + b
print(out)