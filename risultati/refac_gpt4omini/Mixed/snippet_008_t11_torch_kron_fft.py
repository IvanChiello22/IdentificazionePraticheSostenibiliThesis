import torch, torch.fft as fft
A = torch.randn(64, 32); B = torch.randn(32, 16)
K = torch.kron(A, B)
x = torch.randn(8, 1024)
X = fft.rfft(x, dim=-1)
print(K.shape, X.shape)