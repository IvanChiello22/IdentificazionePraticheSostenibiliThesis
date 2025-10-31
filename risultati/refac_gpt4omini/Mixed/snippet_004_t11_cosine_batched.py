A = np.random.randn(500, 128)
B = np.random.randn(600, 128)
A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
S = np.dot(A, B.T)
print(S.shape, round(S.max(),4))