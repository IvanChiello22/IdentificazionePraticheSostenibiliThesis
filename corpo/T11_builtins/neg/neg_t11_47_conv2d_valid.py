# 47) Convoluzione 2D (valid) naive
image = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9,10,11,12],
    [13,14,15,16]
]
kernel = [
    [1, 0],
    [0,-1]
]
H = len(image); W = len(image[0])
kH = len(kernel); kW = len(kernel[0])
out = [[0]*(W-kW+1) for _ in range(H-kH+1)]
for i in range(H-kH+1):
    for j in range(W-kW+1):
        s = 0
        for ki in range(kH):
            for kj in range(kW):
                s += image[i+ki][j+kj] * kernel[ki][kj]
        out[i][j] = s
print(out)
