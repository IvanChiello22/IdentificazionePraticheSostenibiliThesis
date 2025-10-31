import numpy as np
x = np.random.randn(20000)
k = np.hanning(512)

def _next_fft_len(target):
    try:
        return np.fft.next_fast_len(target)
    except AttributeError:
        return 1 << ((target - 1).bit_length())

n = len(x) + len(k) - 1
nfft = _next_fft_len(n)
y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(k, nfft), nfft)[:n]
print(y.shape)