# T11 POS — snippet 48
import numpy as np
A=np.random.randn(5000,64);B=np.random.randn(3000,64)
D2=(A*A).sum(1,keepdims=True)+(B*B).sum(1)[None,:]-2*A@B.T
D=np.sqrt(np.maximum(D2,0.0))
print(D.shape)
