import numpy as np
x=np.random.randn(20000);k=np.hanning(512)
n=len(x)+len(k)-1
_next_fast_len = getattr(np.fft, "next_fast_len", None)
nfft = _next_fast_len(n) if _next_fast_len is not None else 1<<((n-1).bit_length())
y=np.fft.irfft(np.fft.rfft(x,nfft)*np.fft.rfft(k,nfft),nfft)[:n]
print(y.shape)