import numpy as np
import os

A=np.random.randn(5000,64);B=np.random.randn(3000,64)
checkpoint='D_checkpoint.npy'
if os.path.exists(checkpoint):
    try:
        D = np.load(checkpoint)
        if D.shape != (A.shape[0], B.shape[0]):
            raise ValueError("Checkpoint shape mismatch")
    except Exception:
        D2=(A*A).sum(1,keepdims=True)+(B*B).sum(1)[None,:]-2*A@B.T
        D=np.sqrt(np.maximum(D2,0.0))
        try:
            np.save(checkpoint, D)
        except Exception:
            pass
else:
    D2=(A*A).sum(1,keepdims=True)+(B*B).sum(1)[None,:]-2*A@B.T
    D=np.sqrt(np.maximum(D2,0.0))
    try:
        np.save(checkpoint, D)
    except Exception:
        pass
print(D.shape)