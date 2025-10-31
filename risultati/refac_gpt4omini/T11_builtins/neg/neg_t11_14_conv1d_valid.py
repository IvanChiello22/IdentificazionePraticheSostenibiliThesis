signal = [1, 2, 3, 4, 5]
kernel = [1, 0, -1]

def convolve_1d(signal, kernel):
    output_length = len(signal) - len(kernel) + 1
    out = [sum(signal[i + k] * kernel[k] for k in range(len(kernel))) for i in range(output_length)]
    return out

result = convolve_1d(signal, kernel)
print(result)