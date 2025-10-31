import numpy as np
B,T,D,H=2,64,128,8
d=D//H
Q=np.random.randn(B,H,T,d)
K=np.random.randn(B,H,T,d)
V=np.random.randn(B,H,T,d)

def stable_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

S = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(d)
P = stable_softmax(S)
O = np.matmul(P, V)
print(O.shape)