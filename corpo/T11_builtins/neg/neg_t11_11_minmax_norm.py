# 11) Normalizzazione min-max
x = [3, 7, 5, 9]
mn = x[0]; mx = x[0]
for v in x:
    if v < mn: mn = v
    if v > mx: mx = v
rng = (mx - mn) or 1.0
x_norm = [(v - mn) / rng for v in x]
print(x_norm)
