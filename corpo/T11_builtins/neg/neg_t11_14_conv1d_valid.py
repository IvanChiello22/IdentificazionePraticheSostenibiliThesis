# 14) Convoluzione 1D (valid)
signal = [1, 2, 3, 4, 5]
kernel = [1, 0, -1]
out = []
for i in range(len(signal) - len(kernel) + 1):
    s = 0
    for k in range(len(kernel)):
        s += signal[i+k] * kernel[k]
    out.append(s)
print(out)
