# 14) Convoluzione 1D (valid)
signal = [1, 2, 3, 4, 5]
kernel = [1, 0, -1]
out = []
for i in range(len(signal) - len(kernel) + 1):
    out.append(sum(x * y for x, y in zip(signal[i:], kernel)))
print(out)