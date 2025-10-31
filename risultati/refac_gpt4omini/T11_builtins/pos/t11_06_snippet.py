import numpy as np
x = np.random.randn(20000)
k = np.hanning(512)
n = len(x) + len(k) - 1
nfft = 1 << ((n - 1).bit_length())
rfft_x = np.fft.rfft(x, nfft)
rfft_k = np.fft.rfft(k, nfft)
y = np.fft.irfft(rfft_x * rfft_k, nfft)[:n]
print(y.shape)