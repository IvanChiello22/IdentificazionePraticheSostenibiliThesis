# T11 POS — snippet 22
import numpy as np
x=np.random.randn(20000);k=np.hanning(512)
n=len(x)+len(k)-1;nfft=1<<((n-1).bit_length())
y=np.fft.irfft(np.fft.rfft(x,nfft)*np.fft.rfft(k,nfft),nfft)[:n]
print(y.shape)
