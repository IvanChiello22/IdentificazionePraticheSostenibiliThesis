import numpy as np
B,T,D,H=2,64,128,8
d=D//H
Q=np.random.randn(B,H,T,d);K=np.random.randn(B,H,T,d);V=np.random.randn(B,H,T,d)

def _softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)

BH = B * H
Qb = Q.reshape(BH, T, d)
Kb = K.reshape(BH, T, d)
Vb = V.reshape(BH, T, d)

S = np.matmul(Qb, Kb.transpose(0,2,1)) / np.sqrt(d)
S = S.reshape(B, H, T, T)
P = _softmax(S, axis=-1)
P_b = P.reshape(BH, T, T)
O = np.matmul(P_b, Vb).reshape(B, H, T, d)
print(O.shape)