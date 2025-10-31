# T11 POS — snippet 45
import numpy as np
B,T,D,H=2,64,128,8
d=D//H
Q=np.random.randn(B,H,T,d);K=np.random.randn(B,H,T,d);V=np.random.randn(B,H,T,d)
S=np.einsum('bhtd,bhsd->bhts',Q,K)/np.sqrt(d)
P=np.exp(S-S.max(axis=-1,keepdims=True));P/=P.sum(axis=-1,keepdims=True)
O=np.einsum('bhts,bhsd->bhtd',P,V)
print(O.shape)
