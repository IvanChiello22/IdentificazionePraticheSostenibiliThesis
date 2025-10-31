# 29) Layer lineare (manuale)
inp = [1.0, 2.0, -1.0]
W = [[0.1, -0.3],
     [0.5,  0.2],
     [0.0,  0.4]]
b = [0.01, -0.02]
out = [0.0, 0.0]
for j in range(len(out)):
    s = b[j]
    s += sum(x * row[j] for x, row in zip(inp, W))
    out[j] = s
print(out)