import numpy as np
B,T,D,H=2,64,128,8
d=D//H
Q=np.random.randn(B,H,T,d);K=np.random.randn(B,H,T,d);V=np.random.randn(B,H,T,d)

def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(d)
P = _softmax(S, axis=-1)
O = np.matmul(P, V)

try:
    np.save('checkpoint_S.npy', S)
    np.save('checkpoint_O.npy', O)
except Exception:
    pass

print(O.shape)